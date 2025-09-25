# Windmill Ansible Control

This Ansible setup allows you to remotely manage Windmill workers across your distributed network without SSH-ing into individual nodes.

## Quick Start

1. **Update inventory** - Edit `inventory.yml` with your actual node IPs and worker assignments
2. **Test connectivity** - `ansible all -m ping`
3. **Check status** - `./windmill-ctl status`
4. **Control workers** - `./windmill-ctl start|stop|restart [worker]`

## Commands

### Basic Operations
```bash
# Check status of all workers on all nodes
./windmill-ctl status

# Full deployment: stop → update → start (recommended)
./windmill-ctl deploy

# Deploy specific branch
./windmill-ctl deploy --branch dev

# Manual operations
./windmill-ctl update                    # Update code on all nodes (git pull)
./windmill-ctl start                     # Start all workers on all nodes
./windmill-ctl start blip                # Start specific worker on all nodes that support it
```

### Group-Based Control
```bash
# Deploy only to GPU nodes
./windmill-ctl deploy --group gpu_nodes

# Deploy dev branch to light processing nodes
./windmill-ctl deploy --branch dev --group light_processing

# Manual group operations
./windmill-ctl update --group gpu_nodes              # Update code only on GPU nodes
./windmill-ctl start ollama --group ollama_nodes     # Start ollama on its 4 dedicated nodes
./windmill-ctl start colors_post --group light_processing  # Start colors_post on all 7 light processing nodes
./windmill-ctl stop --group gpu_nodes                # Stop all GPU-intensive workers
./windmill-ctl status --group specialized_nodes      # Check status of specialized workers only
```

### Host-Specific Control
```bash
# Stop all workers on specific node
./windmill-ctl stop --host k3.local

# Restart specific worker on specific node
./windmill-ctl restart yolov8 --host k3.local
```

### Topology Overview
```bash
# View current node groups and assignments
./show-topology
```

## Setup

### 1. Install Ansible
```bash
# Ubuntu/Debian
sudo apt install ansible

# macOS
brew install ansible
```

### 2. Configure SSH Keys
Ensure passwordless SSH access to all worker nodes with their respective usernames:
```bash
# Generate key if needed
ssh-keygen -t rsa

# Copy to each worker node with correct username
ssh-copy-id pi@192.168.0.121
ssh-copy-id ubuntu@192.168.0.123
ssh-copy-id admin@192.168.0.124
# etc... (check inventory.yml for each node's username)
```

### 3. Update Inventory
Edit `inventory.yml` to match your network:
- Update IP addresses
- Update hostnames
- Assign worker types to appropriate nodes
- Verify SSH user and paths

### 4. Test Connection
```bash
cd ansible/
ansible all -m ping
```

## Files

- **`inventory.yml`** - Defines your worker nodes and assignments
- **`windmill.yml`** - Ansible playbook that executes windmill.sh remotely
- **`windmill-ctl`** - Convenient wrapper script for common operations
- **`ansible.cfg`** - Ansible configuration

## Architecture Integration

This setup integrates with your existing Pi cluster architecture:
- **k1.local** (192.168.0.121): PostgreSQL + consensus_worker
- **k2.local** (192.168.0.122): RabbitMQ only
- **k3.local** (192.168.0.123): GPU services (blip, clip, yolov8)

The Ansible setup respects your existing `windmill.sh` script and worker assignments, just executing them remotely.