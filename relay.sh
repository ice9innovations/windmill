source windmill_venv/bin/activate

export IMAGE_RELAY_BIND_HOST=192.168.0.101
export IMAGE_RELAY_PORT=8787
export IMAGE_RELAY_ALLOWED_CIDRS=192.168.0.0/16,127.0.0.0/8

python3 image_relay.py
