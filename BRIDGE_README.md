# Keygrinder Bridge to Redis Pub/Sub

This bridge service connects the CUDA keygrinder to the WEWE Vanity Key Service via Redis pub/sub.

## Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Build the keygrinder (if not already built):
```bash
export PATH=/usr/local/cuda/bin:$PATH
make -j$(nproc)
```

## Usage

### Option 1: Run bridge directly (recommended)

The bridge will spawn the keygrinder automatically:

```bash
# Using default settings
npm run bridge

# Or with custom Redis URL
REDIS_URL=redis://your-redis-host:6379 npm run bridge

# Or with custom keygrinder command
KEYGRINDER_CMD=./src/release/cuda_ed25519_vanity npm run bridge
```

### Option 2: Pipe keygrinder output to bridge

If you want to run the keygrinder separately and pipe its output:

```bash
# Run keygrinder and pipe to bridge
LD_LIBRARY_PATH=./src/release ./src/release/cuda_ed25519_vanity | npm run bridge
```

However, the bridge is designed to spawn the keygrinder itself, so Option 1 is recommended.

## How It Works

1. The bridge spawns the keygrinder process and monitors its stdout
2. When a vanity key is found, the keygrinder outputs:
   - A `MATCH` line with the address and seed hex
   - A JSON array line with the keypair format `[seed bytes..., public key bytes...]`
3. The bridge parses these lines and publishes the keypair to Redis pub/sub channel `keypair:requests`
4. The WEWE Vanity Key Service receives the keypair and caches it

## Environment Variables

- `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379`)
- `KEYGRINDER_CMD`: Path to keygrinder executable (default: `./src/release/cuda_ed25519_vanity`)
- `KEYGRINDER_ARGS`: Space-separated arguments to pass to keygrinder (optional)

## Integration with Docker Compose

You can add the bridge to your docker-compose setup. Here's an example addition to the WEWE-Vanity-Key-Service docker-compose.yml:

```yaml
  keygrinder-bridge:
    build:
      context: ../solanity-pub-sub
      dockerfile: Dockerfile.bridge
    environment:
      - REDIS_URL=redis://redis:6379
      - KEYGRINDER_CMD=./src/release/cuda_ed25519_vanity
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ../solanity-pub-sub/src/release:/app/src/release
    restart: unless-stopped
```

Note: The keygrinder requires CUDA/GPU access, so you'll need to configure Docker with GPU support if running in a container.

