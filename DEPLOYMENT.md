# Windmill Deployment Guide

This document is the operator-facing deployment guide for Windmill in the context of a full Animal Farm stack.

It is intentionally broader than [INSTALLATION.md](/home/sd/windmill/INSTALLATION.md): installation explains how to bring up this repo, while this guide explains how Windmill fits into the surrounding system and how to operate it in real deployments.

## Stack Overview

Windmill is the queue-driven orchestration layer for Animal Farm image analysis.

In a typical deployment there are three logical surfaces:

1. API
   - Receives images from clients
   - Validates, normalizes, and registers `images`
   - Publishes primary jobs to RabbitMQ
   - Reads progressive and final results from PostgreSQL
2. Windmill
   - Runs workers that consume RabbitMQ queues
   - Calls ML service endpoints over HTTP
   - Writes primary, aggregate, and postprocessing results to PostgreSQL
   - Tracks lifecycle state in `service_dispatch` and worker liveness in `worker_registry`
3. UI
   - Talks to the API only
   - Never talks directly to PostgreSQL or RabbitMQ

Windmill does not host the ML models itself. It expects one or more Animal Farm-compatible services to be reachable at the hosts and ports declared in `service_config.yaml`.

## Responsibilities By Component

### Windmill

- Queue consumers and worker orchestration
- Primary result persistence in `results`
- Downstream trigger routing
- Harmonization, consensus, noun/verb consensus, caption summary, grounding, and content analysis
- Operational lifecycle tracking

### Upstream API

- Owns client authentication and request admission
- Mints `image_id`
- Publishes primary queue messages with a stable `trace_id`
- Exposes status/results endpoints to clients

### UI

- Polls or streams status from the API
- Renders lifecycle and result state
- Should treat the API as its only backend

## Core Infrastructure

Windmill requires:

- PostgreSQL 13+ with `pgvector`
- RabbitMQ 3.x with durable queues
- Python 3.10+
- Animal Farm-compatible HTTP services implementing `/analyze`-style endpoints

Optional but common:

- systemd for API/worker supervision
- Ansible for multi-node control via [ansible/README.md](/home/sd/windmill/ansible/README.md)

## Typical Topologies

### Single Host

Suitable for development or small deployments.

- PostgreSQL on the same machine
- RabbitMQ on the same machine
- Windmill API on the same machine
- Windmill workers on the same machine
- ML services on the same machine

### Split Infra

Common production pattern.

- PostgreSQL on a database host
- RabbitMQ on a broker host
- Windmill API on an app host
- Windmill workers on one or more compute nodes
- ML services on the same compute nodes or on dedicated inference hosts

### Distributed Workers

Windmill is designed to run multiple worker copies across machines.

- Any node can run any subset of workers
- Multiple copies of the same worker are generally allowed
- Shared coordination happens through RabbitMQ and PostgreSQL
- `worker_registry` is the live view of online workers

## Network Requirements

The table below documents the network flows Windmill depends on.

| From | To | Port | Purpose | Public? |
|---|---|---:|---|---|
| Windmill API | PostgreSQL | 5432 | image registration and result reads | No |
| Windmill API | RabbitMQ | 5672/5671 | primary job publish | No |
| Windmill workers | PostgreSQL | 5432 | results, dispatch, worker registry | No |
| Windmill workers | RabbitMQ | 5672/5671 | consume, publish, DLQ | No |
| Windmill workers | ML services | service-specific | `/analyze` calls | Usually no |
| UI | API | deployment-specific | user-facing application traffic | Yes, via proxy |
| Operators | RabbitMQ management UI | 15672 | ops/debugging | No, VPN/admin only |

What should not be publicly reachable:

- PostgreSQL
- RabbitMQ broker ports
- RabbitMQ management UI
- Internal ML service ports
- Windmill API directly, if it is normally fronted by another API or reverse proxy

## First-Time Setup Sequence

Use this order for a new environment:

1. Provision PostgreSQL and RabbitMQ.
2. Clone Windmill and create `.env`.
3. Apply [db/db.sql](/home/sd/windmill/db/db.sql).
4. Load ConceptNet with `utils/load_conceptnet.sh` if noun consensus is needed.
5. Configure `service_config.yaml` to point at real ML service endpoints.
6. Start the upstream API that will publish primary Windmill jobs.
7. Start Windmill workers.
8. Submit a single test image and verify:
   - `images` row exists
   - primary `results` rows appear
   - `service_dispatch` converges
   - downstream aggregate tables populate as expected

## Windmill Deployment Notes

### Configuration Sources

- `.env`
  - queue connectivity
  - database connectivity
  - worker timeouts / logging / prefetch defaults
- `service_config.yaml`
  - service catalog
  - queue names
  - service types
  - tier membership
  - operator-owned routing targets
- [docs/workflow.md](/home/sd/windmill/docs/workflow.md)
  - documentation-only internal dependency map
  - documents which downstream stages are expected and which worker path triggers them

### Runtime Entry Points

- `api.py`
  - thin local submission/status API for Windmill itself
  - also exposes `GET /workflow` for machine-readable dependency metadata
- `workers/*.py`
  - worker entry points
- `./windmill.sh`
  - local worker lifecycle management
- `ansible/windmill-ctl`
  - multi-node worker control

### systemd Pattern

Common production pattern:

- One systemd unit for the API
- Windmill workers managed either by:
  - `./windmill.sh` under a process supervisor
  - separate systemd units
  - Ansible-driven control using the repo's existing wrapper scripts

The exact unit files are deployment-specific, but verification is always the same:

- `systemctl status <service>`
- `./windmill.sh status`
- `SELECT * FROM worker_registry WHERE status = 'online'`

## Operational Verification

After deployment, verify all of the following:

### Database

- `images` accepts inserts
- `results` rows are being written
- `service_dispatch` moves through `pending -> complete/failed/dead-lettered`
- `worker_registry` heartbeats are fresh

### RabbitMQ

- primary queues have consumers
- downstream queues have consumers
- DLQs are empty or explainable

### Windmill

- `./windmill.sh status` matches expected local workers
- `logs/` is updating
- no worker is crash-looping

### End-to-End

Submit one image and verify:

- primary services expected for the tier run
- `trace_id` is present on primary `results.source_trace_id`
- downstream stages expected for the tier converge
- API status eventually reports completion

## Multi-Node Operations

For distributed deployments:

- keep `.env` consistent across nodes that share the same DB and broker
- let `service_config.yaml` reflect reachable service endpoints from each worker node
- use [ansible/inventory.example.yml](/home/sd/windmill/ansible/inventory.example.yml) as the starting point for node assignment
- avoid running duplicate singleton-style operational helpers unless they are designed for it

Windmill's normal worker model assumes horizontal fan-out is acceptable. If a worker cannot safely run in parallel, document that explicitly in your deployment inventory.

## Troubleshooting Entry Points

Start here when something is wrong:

- [RUNBOOK.md](/home/sd/windmill/RUNBOOK.md) for DLQ triage and reprocessing patterns
- [README.md](/home/sd/windmill/README.md) for schema and lifecycle overview
- [INSTALLATION.md](/home/sd/windmill/INSTALLATION.md) for first-time setup assumptions
- [ansible/README.md](/home/sd/windmill/ansible/README.md) for multi-node control

## Scope Boundary

This repo documents Windmill's role in the stack. It cannot fully document installation of every upstream API, UI, or ML service implementation because those may live in separate repos or private infrastructure.

The contract Windmill assumes is simple:

- an upstream API or producer publishes primary jobs with `image_id`, `image_data`, `tier`, and `trace_id`
- Animal Farm-compatible services accept image requests at the configured endpoints
- PostgreSQL and RabbitMQ are reachable from every Windmill node
