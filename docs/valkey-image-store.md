# Valkey Image Store

Windmill supports two image transport modes:

- `inline`: embed image bytes in RabbitMQ messages
- `valkey`: store image bytes in Valkey and pass opaque refs through RabbitMQ

`inline` is the default. Use `valkey` when you want uploaded images and bbox
crops to remain ephemeral and avoid being written to disk or carried through
queue payloads.

## Required `.env` settings

```bash
IMAGE_STORE_MODE=valkey
VALKEY_HOST=images.ice9.ai
VALKEY_PORT=6379
VALKEY_SSL=true
VALKEY_USERNAME=windmill
VALKEY_PASSWORD=...
VALKEY_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
VALKEY_IMAGE_TTL_SECONDS=90
VALKEY_CROP_TTL_SECONDS=90
VALKEY_SOCKET_CONNECT_TIMEOUT_SECONDS=3
VALKEY_SOCKET_TIMEOUT_SECONDS=3
VALKEY_HEALTH_CHECK_INTERVAL_SECONDS=30
VALKEY_KEEPALIVE_PING_SECONDS=15

# Optional on same-LAN workers only
IMAGE_RELAY_URL=
IMAGE_RELAY_TIMEOUT_SECONDS=5
IMAGE_RELAY_FALLBACK_DIRECT=false
```

## Expected Valkey server properties

- TLS enabled
- ACL auth enabled
- persistence disabled:
  - `save ""`
  - `appendonly no`
- recommended memory policy:
  - `maxmemory-policy noeviction`

## Smoke test from a Windmill node

```bash
export REDISCLI_AUTH='your-secret'
valkey-cli --tls -h "$VALKEY_HOST" -p "$VALKEY_PORT" \
  --cacert "$VALKEY_CA_CERTS" \
  --user "$VALKEY_USERNAME" \
  ping
unset REDISCLI_AUTH
```

Expected result:

```text
PONG
```

## Runtime behavior

When `IMAGE_STORE_MODE=valkey`:

- `api.py` stores uploaded image bytes in Valkey and publishes `image_ref`
- primary workers resolve `image_ref` before calling ML services
- downstream system queues forward `image_ref` only where the consumer still
  needs original image bytes
- harmony bbox postprocessing stores crop bytes in Valkey and publishes
  `crop_ref`

When `IMAGE_STORE_MODE=inline`:

- Windmill keeps the legacy behavior and publishes inline base64 image payloads

## Rollout notes

- Restart the API and all workers after changing `IMAGE_STORE_MODE`
- Keep `VALKEY_IMAGE_TTL_SECONDS` and `VALKEY_CROP_TTL_SECONDS` at `90` unless
  you have measured queue latency comfortably below that
- Keep Valkey socket timeouts enabled so a stale image-store connection fails
  fast instead of wedging a worker on `get_image()`
- Remote idle TLS connections can still go stale. Windmill retries one fetch
  after dropping the client pool on Valkey timeout/connection errors so cold
  reconnects are bounded by the configured socket timeouts instead of the
  previous longer stall
- Windmill now also sends periodic Valkey `PING`s from each process when
  `VALKEY_KEEPALIVE_PING_SECONDS` is set, so infrequently used workers do not
  pay the first-request reconnect penalty as often
- Same-site workers can read through a RAM-only relay by setting
  `IMAGE_RELAY_URL`. Leave it unset for hosts at other sites, such as Boden.
- A Valkey miss should be treated as an infrastructure problem, not a normal
  model failure

## Dorothy LAN image relay

Start one relay on Dorothy with the normal Valkey credentials in its environment:

```bash
IMAGE_STORE_MODE=valkey \
VALKEY_HOST=images.ice9.ai \
IMAGE_RELAY_BIND_HOST=192.168.0.101 \
IMAGE_RELAY_PORT=8787 \
IMAGE_RELAY_ALLOWED_CIDRS=192.168.0.0/16 \
python image_relay.py
```

The relay exposes:

- `GET /healthz`
- `GET /metrics`
- `GET /image/<image_ref>?kind=image`

The cache is RAM only. Size and lifetime are controlled by:

- `IMAGE_RELAY_CACHE_TTL_SECONDS` default `15`
- `IMAGE_RELAY_MAX_CACHE_BYTES` default `536870912`
- `IMAGE_RELAY_MAX_OBJECT_BYTES` default `67108864`
- `IMAGE_RELAY_WAITER_TIMEOUT_SECONDS` default `30`

The effective relay cache TTL is capped at 80% of the backing Valkey image or
crop TTL, so relay cache hits cannot keep serving bytes until the upstream ref
has already expired.

Roll out manually:

1. Start the relay on Dorothy.
2. Verify health from Dorothy: `curl http://192.168.0.101:8787/healthz`.
3. Verify LAN reachability from one Orin and confirm non-LAN hosts cannot reach it.
4. Configure only one Orin worker first, preferably NSFW2:
   `IMAGE_RELAY_URL=http://192.168.0.101:8787` and
   `IMAGE_RELAY_FALLBACK_DIRECT=true`.
5. Restart only that worker.
6. Compare `/metrics`, relay logs, and existing
   `image_fetch_duration_seconds` event timings.
7. Expand gradually to the remaining home-LAN workers.
8. Leave Boden direct, or give Boden its own local relay later.

## Current coverage

Valkey mode is wired into:

- API `/analyze`
- primary workers
- noun-consensus-triggered Florence grounding
- rembg
- harmony bbox postprocessing
- producer and retrigger utilities

Legacy inline fallbacks remain in place so old queue messages can drain safely.
