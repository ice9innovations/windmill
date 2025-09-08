## Windmill Runbook

### 1) Restart Order
- Stop all: `./windmill.sh stop` (or `./windmill.sh stop all`)
- Start infra (if external, ensure reachable): RabbitMQ, Postgres
- Start workers: `./windmill.sh start`
- Verify: `./windmill.sh status`

### 2) Common Failures and Fixes
- No jobs flowing
  - Check RabbitMQ UI (queue depths)
  - Ensure producer submitted jobs
  - Verify workers connected (logs/)
- Repeated errors on a queue
  - Increase log verbosity: `LOG_LEVEL=DEBUG`
  - Inspect DLQ for that queue (README: DLQ section)
  - Fix root cause, then requeue a small batch from DLQ
- Database connection errors
  - Verify DB env vars
  - Check DB reachability from container/host

### 3) DLQ Triage
- List DLQs: `python utils/list_dlqs.py`
- Sample messages via UI (Get messages)
- Purge only if data is irrecoverable; otherwise requeue a small limit: `python utils/requeue_from_dlq.py <queue> --limit 100`

### 4) Reprocessing
- Harmony changes (merged boxes):
  - Re-run harmony over affected images, then rerun consensus
  - Optionally script selective requeue to `queue_harmony`
- Consensus changes:
  - Re-run consensus by publishing to the consensus queue for affected images

### 5) Observability Quick Tips
- Logs in `logs/`
- RabbitMQ UI at `http://<host>:15672`
- Add `LOG_LEVEL=DEBUG` for deeper tracing

### 6) Safe Ops Practices
- Use DLQ instead of infinite requeues
- Requeue in small batches; monitor error rates
- Prefer URL-based datasets to avoid container file mounts


