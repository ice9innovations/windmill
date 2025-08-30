# status_workers.sh
#!/bin/bash
for pidfile in pids/*.pid; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        SERVICE=$(basename "$pidfile" .pid)
        if kill -0 $PID 2>/dev/null; then
            echo "✅ $SERVICE is running (PID: $PID)"
        else
            echo "❌ $SERVICE is dead (PID: $PID)"
            rm "$pidfile"
        fi
    fi
done
