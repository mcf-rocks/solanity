# How to Run Solanity Keygrinder + Bridge

## Overview

You need **Docker** for Redis and the API service, but the keygrinder and bridge run **locally** (outside Docker) because:
- The keygrinder requires **CUDA/GPU access** which is complex in Docker
- Running locally is simpler and more performant for GPU workloads

## Architecture

```
┌─────────────────┐
│   Docker        │
│  ┌───────────┐  │
│  │  Redis    │  │◄───┐
│  └───────────┘  │    │
│  ┌───────────┐  │    │ Redis Pub/Sub
│  │  API      │  │    │
│  │ (Express) │  │    │
│  └───────────┘  │    │
└─────────────────┘    │
                        │
┌─────────────────┐    │
│   Local Host    │    │
│  ┌───────────┐  │    │
│  │ Keygrinder│──┼────┘
│  │  (CUDA)   │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Bridge   │──┼────┐
│  │ (Node.js) │  │    │
│  └───────────┘  │    │
└─────────────────┘    │
                        │
                        ▼
                  Redis Pub/Sub
```

## Step-by-Step Instructions

### Prerequisites

1. **Docker & Docker Compose** - For Redis and API service
2. **CUDA Toolkit** - For building/running the keygrinder
3. **NVIDIA GPU** - Required for the keygrinder
4. **Node.js** - For the bridge service

### Step 1: Start Redis and API Service (Docker)

**Important:** Start Docker services FIRST, then run the bridge. The bridge will wait for Redis to be ready.

```bash
cd WEWE-Vanity-Key-Service
docker-compose up -d
```

This starts:
- Redis on `localhost:6379` (starts first)
- API service on `localhost:3001` (waits for Redis to be healthy, port 3001 to avoid conflicts)

Wait a few seconds for services to start, then verify:
```bash
# Check Redis
redis-cli ping  # Should return PONG

# Check API
curl http://localhost:3001/health
```

### Step 2: Build the Keygrinder (Local)

```bash
cd solanity-pub-sub

# Make sure CUDA is in your PATH
export PATH=/usr/local/cuda/bin:$PATH

# Build the keygrinder
make -j$(nproc)
```

This creates: `./src/release/cuda_ed25519_vanity`

**Optional:** Configure vanity patterns in `src/config.h` before building.

### Step 3: Install Bridge Dependencies (Local)

```bash
cd solanity-pub-sub
npm install
```

### Step 4: Run the Bridge (Local)

**The bridge will automatically wait for Redis to be available** (up to 60 seconds), so you can start it right after Docker Compose.

The bridge automatically spawns the keygrinder and connects to Redis:

```bash
# From solanity-pub-sub directory
npm run bridge
```

Or with explicit Redis URL:
```bash
REDIS_URL=redis://localhost:6379 npm run bridge
```

The bridge will:
1. **Wait for Redis** to be available (retries every 2 seconds)
2. Connect to Redis once ready
3. Spawn the keygrinder process
4. Parse keygrinder output for vanity keys
5. Publish found keys to Redis pub/sub
6. The API service will receive and cache them

## Alternative: Run Keygrinder Separately

If you want to run the keygrinder manually:

```bash
# Terminal 1: Run keygrinder
cd solanity-pub-sub
LD_LIBRARY_PATH=./src/release ./src/release/cuda_ed25519_vanity

# Terminal 2: Run bridge (it will read from stdin if keygrinder is piped)
# But the bridge is designed to spawn keygrinder itself, so this isn't recommended
```

## Testing

### Test 1: Check if keys are being generated

Watch the bridge output - you should see:
```
[Bridge] Found match: AAAAA...
[Bridge] ✓ Published keypair abc123... (AAAAA...) to Redis
```

### Test 2: Request keys from API

```bash
curl http://localhost:3001/get-key
```

### Test 3: Check cache status

```bash
curl http://localhost:3001/health | jq
```

### Test 4: Monitor Redis pub/sub

```bash
redis-cli
> PSUBSCRIBE keypair:*
```

## Troubleshooting

### "Cannot connect to Redis"
- Make sure Docker Compose is running: `docker-compose ps`
- Check Redis is accessible: `redis-cli -h localhost -p 6379 ping`
- Verify `REDIS_URL` environment variable

### "Keygrinder not found"
- Make sure you built it: `ls -la src/release/cuda_ed25519_vanity`
- Check CUDA is installed: `nvcc --version`
- Verify GPU is accessible: `nvidia-smi`

### "No keys appearing in API"
- Check bridge logs for `[Bridge] ✓ Published` messages
- Verify API service is running: `docker-compose logs api`
- Check Redis connection: `docker-compose logs redis`

## Why Not Docker for Keygrinder?

The keygrinder **could** run in Docker, but it requires:
1. **NVIDIA Container Toolkit** (`nvidia-docker`)
2. **GPU passthrough** configuration
3. **CUDA libraries** in the container
4. More complex setup

Running locally is:
- ✅ Simpler setup
- ✅ Better GPU performance
- ✅ Easier debugging
- ✅ No Docker GPU configuration needed

## Summary

**Docker Required:** ✅ Yes, for Redis and API service  
**Docker for Keygrinder:** ❌ No, runs locally  
**Docker for Bridge:** ❌ Optional, but runs locally for simplicity

**Quick Start:**
```bash
# Terminal 1: Start services
cd WEWE-Vanity-Key-Service && docker-compose up -d

# Terminal 2: Run bridge (builds & runs keygrinder automatically)
cd solanity-pub-sub && npm install && npm run bridge
```

