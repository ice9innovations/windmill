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
VALKEY_HOST=valkey.example.com
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
export REDISCLI_AUTH="$VALKEY_PASSWORD"
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
  `IMAGE_RELAY_URL`. Leave it unset for hosts at other sites.
- A Valkey miss should be treated as an infrastructure problem, not a normal
  model failure

## LAN image relay

Start one relay on the same LAN as latency-sensitive workers with the normal
Valkey credentials in its environment:

```bash
IMAGE_STORE_MODE=valkey \
VALKEY_HOST=valkey.example.com \
IMAGE_RELAY_BIND_HOST=198.51.100.10 \
IMAGE_RELAY_PORT=8787 \
IMAGE_RELAY_ALLOWED_CIDRS=198.51.100.0/24 \
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

For a healthy simultaneous burst, `/metrics` should show one `cache_misses` /
`upstream_fetches` leader for that ref and the rest as `singleflight_waiters`.
For example, five concurrent service requests for one image should look like
`requests=5`, `cache_misses=1`, `upstream_fetches=1`,
`singleflight_waiters=4`.

Roll out manually:

1. Start the relay on the LAN host.
2. Verify health locally: `curl http://198.51.100.10:8787/healthz`.
3. Verify LAN reachability from one worker and confirm non-LAN hosts cannot reach it.
4. Configure only one same-LAN worker first, preferably NSFW2:
   `IMAGE_RELAY_URL=http://198.51.100.10:8787` and
   `IMAGE_RELAY_FALLBACK_DIRECT=true`.
5. Restart only that worker.
6. Compare `/metrics`, relay logs, and existing
   `image_fetch_duration_seconds` event timings.
7. Expand gradually to the remaining home-LAN workers.
8. Leave remote-site workers direct, or give each site its own local relay later.

## Current coverage

Valkey mode is wired into:

- API `/analyze`
- primary workers
- noun-consensus-triggered Florence grounding
- rembg
- harmony bbox postprocessing
- producer and retrigger utilities

Legacy inline fallbacks remain in place so old queue messages can drain safely.
