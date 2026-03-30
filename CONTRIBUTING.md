# Contributing To Windmill

This guide is for contributors extending Windmill or adding Animal Farm-compatible services that Windmill can orchestrate.

It is intentionally focused on standard extension paths:

- add a new ML service that speaks the expected HTTP contract
- add a new Windmill worker that uses existing pipeline patterns

If your change requires a new pipeline stage, new lifecycle semantics, or cross-repo product behavior, expect to read more code than this guide covers.

## Two Contribution Surfaces

### 1. Add An Animal Farm-Compatible ML Service

Windmill calls services over HTTP. A service can live in another repo or be maintained independently as long as it behaves like an Animal Farm service.

At minimum it should:

- accept image input at the configured endpoint
- return JSON in the shape the corresponding worker expects
- return meaningful HTTP errors on failure
- document hardware/runtime requirements

Windmill does not infer schemas dynamically. A new service needs a worker that knows how to interpret the response.

### 2. Add A Windmill Worker

A worker connects a queue to a service and decides:

- how to call the service
- how to store results
- whether to trigger downstream work
- how lifecycle rows are updated

Most primary workers are thin wrappers around [workers/base_worker.py](/home/sd/windmill/workers/base_worker.py). Start there before writing anything custom.

## Service Taxonomy

`service_config.yaml` uses `service_type` tags to drive behavior.

Current important types:

- `spatial`
  - full-image bbox detectors
  - can trigger `harmony`
- `semantic`
  - caption/description-style services
- `vlm`
  - semantic services whose outputs feed noun/verb consensus
- `specialized`
  - detectors or analyzers with service-specific semantics
- `colors`
  - color extraction services
- `metadata`
  - image metadata services
- `grounding`
  - noun-triggered grounding stage
- internal system types such as `harmonization`, `caption_summary`, `content_analysis`, `rembg`

`consensus: true` is opt-in. A primary service only contributes to consensus if that flag is explicitly set.

## Adding A New Primary Or Postprocessing Service

### Step 1: Add It To `service_config.yaml`

Every service must be declared in [service_config.yaml](/home/sd/windmill/service_config.yaml) with:

- category: `primary`, `postprocessing`, or `system`
- `queue_name`
- `host`
- `port`
- `endpoint`
- `service_type`
- `tier`
- noun/verb consensus-derived downstream stages where appropriate

Choose the category based on input shape:

- `primary`
  - runs on the full submitted image
- `postprocessing`
  - runs on merged bbox crops
- `system`
  - internal stage, not a model-serving edge

### Step 2: Create The Worker

Use the naming convention:

- `workers/<service>_worker.py`

That keeps it discoverable by `windmill.sh`.

For simple primary services, the worker is usually just:

```python
from base_worker import BaseWorker

class MyServiceWorker(BaseWorker):
    def __init__(self):
        super().__init__('primary.my_service')
```

If the service accepts the standard image payload and the default result-write path is sufficient, `BaseWorker` does most of the work.

### Step 3: Decide Whether `BaseWorker` Is Enough

`BaseWorker` already handles:

- DB and RabbitMQ connection setup
- worker registry heartbeats
- primary result writes to `results`
- `trace_id` persistence in `results.source_trace_id`
- idempotent primary writes on retry
- synchronous broker-confirmed primary downstream publishes
- ack/nack safety for the common primary path

Use a custom `process_message()` only when the service needs non-standard behavior.

Examples already in the repo:

- [workers/harmony_worker.py](/home/sd/windmill/workers/harmony_worker.py)
- [workers/noun_consensus_worker.py](/home/sd/windmill/workers/noun_consensus_worker.py)

## Dispatch Lifecycle Rules

This is where most subtle bugs happen.

### Primary Workers

Primary workers should rely on `BaseWorker` unless there is a strong reason not to.

The expected flow is:

1. consume one source queue message
2. call the ML service
3. write the primary result
4. mark the primary `service_dispatch` row complete
5. publish required downstream messages with broker confirms
6. ack only after publish succeeds

Do not reintroduce:

- ack-before-publish patterns
- swallowed publish failures
- non-idempotent retry behavior for primary results

### Custom Workers

If you override `process_message()`, every terminal path must do one of:

- `_safe_ack(...)` after success
- `_safe_nack(..., requeue=True)` for retryable failure
- `_safe_nack(..., requeue=False)` only when you intentionally want broker-side DLQ behavior

And every acknowledged path must settle any `service_dispatch` row it created.

Do not ack a message while leaving its dispatch row `pending`.

## Downstream Triggering

The documentation-only trigger map is documented in [docs/workflow.md](/home/sd/windmill/docs/workflow.md).

Current high-level rules:

- spatial primary services may trigger `harmony`
- VLM primary services trigger `noun_consensus` and `verb_consensus`
- `harmony` triggers bbox-level postprocessing
- `noun_consensus` can trigger `florence2_grounding` and `caption_summary`
- `noun_consensus` can also trigger `content_analysis`

If you add a new downstream stage, update both:

- the runtime logic
- [docs/workflow.md](/home/sd/windmill/docs/workflow.md)

## Producer Contract

Primary queue messages are expected to carry:

- `image_id`
- `image_filename`
- `image_data`
- `tier`
- `trace_id`

`trace_id` is important. Windmill now uses it as the stable submission identity for primary-result idempotency.

If you add a new producer path, it must preserve that field.

## Choosing A Service Type

Ask these questions:

- Does this service return bounding boxes on the full image?
  - use `primary` + `spatial`
- Does it describe the whole image in language?
  - use `primary` + `semantic`, and maybe `vlm`
- Does it consume merged bbox crops?
  - use `postprocessing`
- Is it an aggregate or synthesis stage driven by Windmill itself?
  - use `system`

If a primary service should not feed noun/verb consensus-derived stages, set `consensus: false`.

## Testing Expectations

Before submitting a change:

- run `python -m py_compile` on edited workers
- run `bash -n windmill.sh` if you touched the control script
- submit at least one image through the API when the change affects runtime flow
- inspect `service_dispatch` for stuck `pending` rows
- inspect `results` for duplicate primary rows if you changed retry behavior

For lifecycle changes, prefer an explicit fault-injection or stress test over assuming the happy path is enough.

## Documentation Expectations

If you change pipeline behavior, update:

- [README.md](/home/sd/windmill/README.md) for operator-facing overview
- [docs/workflow.md](/home/sd/windmill/docs/workflow.md) if dependency relationships changed
- [INSTALLATION.md](/home/sd/windmill/INSTALLATION.md) if setup requirements changed
- [DEPLOYMENT.md](/home/sd/windmill/DEPLOYMENT.md) if ops or topology assumptions changed
