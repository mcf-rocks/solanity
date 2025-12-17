# Quick Start: Connecting Keygrinder to Vanity Key Service

## Prerequisites

1. **Redis running** - Either locally or via Docker Compose from the WEWE-Vanity-Key-Service project
2. **Keygrinder built** - The CUDA vanity key finder should be compiled
3. **Node.js installed** - For running the bridge service

## Step-by-Step Setup

### 1. Start Redis and the Vanity Key Service

In the `WEWE-Vanity-Key-Service` directory:

```bash
cd ../WEWE-Vanity-Key-Service
docker-compose up -d
```

This starts:
- Redis on `localhost:6379`
- API service on `localhost:3001` (port 3001 to avoid conflicts)

### 2. Install Bridge Dependencies

In the `solanity-pub-sub` directory:

```bash
npm install
```

### 3. Build the Keygrinder (if not already built)

```bash
export PATH=/usr/local/cuda/bin:$PATH
make -j$(nproc)
```

### 4. Run the Bridge

The bridge will automatically spawn the keygrinder and publish found keys to Redis:

```bash
npm run bridge
```

Or with custom Redis URL:

```bash
REDIS_URL=redis://localhost:6379 npm run bridge
```

## Testing the Integration

### Test 1: Check the API receives keys

1. Start the bridge (it will start grinding keys)
2. In another terminal, request a key from the API:

```bash
curl http://localhost:3001/get-key
```

The API should return a keypair. If the keygrinder has found keys and published them, you might get one of those cached keys.

### Test 2: Check health endpoint

```bash
curl http://localhost:3001/health
```

This shows how many keys are cached in Redis.

### Test 3: Monitor Redis pub/sub (optional)

You can use `redis-cli` to monitor the pub/sub channel:

```bash
redis-cli
> PSUBSCRIBE keypair:*
```

You'll see messages when keys are published.

## How It Works

```
┌─────────────┐         ┌──────────┐         ┌──────────────┐
│ Keygrinder  │────────▶│  Bridge   │────────▶│    Redis     │
│  (CUDA)     │ stdout  │ (Node.js) │ pub/sub │   Pub/Sub    │
└─────────────┘         └──────────┘         └──────┬─────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │ Vanity Key   │
                                            │   Service    │
                                            │  (Express)   │
                                            └──────────────┘
```

1. **Keygrinder** finds vanity addresses and outputs them to stdout
2. **Bridge** parses the output and publishes keypairs to Redis pub/sub
3. **Vanity Key Service** receives the keypairs via pub/sub and caches them
4. **API clients** can request keys via HTTP GET `/get-key`

## Troubleshooting

### Bridge can't connect to Redis

- Check Redis is running: `redis-cli ping` should return `PONG`
- Verify `REDIS_URL` environment variable is correct
- Check firewall/network settings

### Keygrinder not starting

- Verify CUDA is installed and accessible
- Check the keygrinder binary exists at `./src/release/cuda_ed25519_vanity`
- Make sure you've built it: `make -j$(nproc)`

### Keys not appearing in API

- Check bridge logs for `[Bridge] ✓ Published` messages
- Verify the Vanity Key Service is running and connected to Redis
- Check Redis pub/sub channel name matches: `keypair:requests`

## Next Steps

- Configure vanity patterns in `src/config.h`
- Adjust `MAX_CACHED_KEYS` in docker-compose.yml
- Scale up by running multiple keygrinder instances (they'll all publish to the same Redis)

