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
- A Valkey miss should be treated as an infrastructure problem, not a normal
  model failure

## Current coverage

Valkey mode is wired into:

- API `/analyze`
- primary workers
- noun-consensus-triggered Florence grounding
- rembg
- harmony bbox postprocessing
- producer and retrigger utilities

Legacy inline fallbacks remain in place so old queue messages can drain safely.
