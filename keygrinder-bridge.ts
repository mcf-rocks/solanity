#!/usr/bin/env ts-node

import { spawn } from 'child_process';
import { createClient } from 'redis';
import readline from 'readline';
import crypto from 'crypto';
import { existsSync } from 'fs';
import { access } from 'fs/promises';
import { constants } from 'fs';

const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const KEYGRINDER_CMD = process.env.KEYGRINDER_CMD || './src/release/cuda_ed25519_vanity';
const KEYGRINDER_ARGS = process.env.KEYGRINDER_ARGS?.split(' ') || [];

async function waitForRedis(publisher: ReturnType<typeof createClient>, maxRetries: number = 30, delayMs: number = 2000): Promise<void> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await publisher.connect();
      await publisher.ping();
      console.log('Connected to Redis');
      return;
    } catch (error) {
      if (i === 0) {
        console.log(`Waiting for Redis at ${REDIS_URL}...`);
      }
      try {
        await publisher.quit();
      } catch {
        // Ignore quit errors
      }
      if (i < maxRetries - 1) {
        process.stdout.write(`  Retry ${i + 1}/${maxRetries}...\r`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      } else {
        throw new Error(`Failed to connect to Redis after ${maxRetries} attempts: ${error}`);
      }
    }
  }
}

async function checkKeygrinderExists(cmd: string): Promise<string> {
  // Check if CUDA keygrinder exists
  if (existsSync(cmd)) {
    try {
      await access(cmd, constants.X_OK);
      return cmd; // Use CUDA version
    } catch {
      // File exists but not executable
    }
  }
  
  // Fallback to CPU version
  const cpuKeygrinder = 'ts-node cpu-keygrinder.ts';
  if (cmd.includes('cuda_ed25519_vanity')) {
    console.log('⚠ CUDA keygrinder not found, using CPU fallback');
    console.log('  Note: CPU version is much slower but works on any system');
    return cpuKeygrinder;
  }
  
  // If custom command was provided and doesn't exist, throw error
  throw new Error(
    `Keygrinder executable not found at: ${cmd}\n` +
    `Please build the CUDA keygrinder first:\n` +
    `  1. Make sure CUDA is installed and in your PATH\n` +
    `  2. Run: export PATH=/usr/local/cuda/bin:$PATH\n` +
    `  3. Run: make -j$(nproc)\n` +
    `  4. Verify: ls -la ${cmd}\n\n` +
    `Or use CPU fallback: npm run bridge:cpu`
  );
}

async function main() {
  console.log('Starting Keygrinder Bridge...');
  console.log(`Redis URL: ${REDIS_URL}`);
  console.log(`Keygrinder Command: ${KEYGRINDER_CMD}`);

  // Check if keygrinder exists before connecting to Redis
  let actualKeygrinderCmd = KEYGRINDER_CMD;
  try {
    actualKeygrinderCmd = await checkKeygrinderExists(KEYGRINDER_CMD);
  } catch (error) {
    console.error('\n❌ Keygrinder check failed:');
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  }

  // Create Redis client and wait for it to be available
  const publisher = createClient({ url: REDIS_URL });
  await waitForRedis(publisher);

  const PUBSUB_CHANNEL = 'keypair:requests';

  // Spawn the keygrinder process (use actual command, which may be CPU fallback)
  let keygrinder: ReturnType<typeof spawn>;
  
  if (actualKeygrinderCmd.startsWith('ts-node')) {
    // CPU fallback - use ts-node
    const scriptPath = actualKeygrinderCmd.replace('ts-node ', '');
    keygrinder = spawn('ts-node', [scriptPath, ...KEYGRINDER_ARGS], {
      stdio: ['inherit', 'pipe', 'inherit'],
      cwd: process.cwd()
    });
  } else {
    // CUDA version - use executable directly
    // Set LD_LIBRARY_PATH to find libcuda-crypt.so
    const libPath = process.cwd() + '/src/release';
    keygrinder = spawn(actualKeygrinderCmd, KEYGRINDER_ARGS, {
      stdio: ['inherit', 'pipe', 'inherit'],
      cwd: process.cwd(),
      env: {
        ...process.env,
        LD_LIBRARY_PATH: `${libPath}:${process.env.LD_LIBRARY_PATH || ''}`.replace(/^:/, '')
      }
    });
  }

  // Create readline interface to parse stdout line by line
  const rl = readline.createInterface({
    input: keygrinder.stdout!,
    crlfDelay: Infinity
  });

  let currentMatch: { address?: string; seedHex?: string } = {};

  rl.on('line', async (line) => {
    const trimmed = line.trim();
    if (!trimmed) return; // Skip empty lines

    // Log initialization and progress messages
    if (trimmed.includes('GPU: Initializing') || 
        trimmed.includes('GPU:') && trimmed.includes('(') ||
        trimmed.includes('Initialising from entropy') ||
        trimmed.includes('END: Initializing') ||
        trimmed.includes('executions') ||
        trimmed.includes('keys found')) {
      console.log(`[Keygrinder] ${trimmed}`);
    }

    // Parse MATCH line: "GPU 0 MATCH AAAA... - abc123..."
    if (trimmed.includes('MATCH')) {
      const match = trimmed.match(/MATCH\s+(\S+)\s+-\s+(\S+)/);
      if (match) {
        currentMatch.address = match[1];
        currentMatch.seedHex = match[2];
        console.log(`[Bridge] Found match: ${currentMatch.address}`);
      }
    }
    // Parse keypair array line: [123,45,67,...]
    else if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        const keypairArray: number[] = JSON.parse(trimmed);
        
        if (keypairArray.length === 64) {
          // Generate a unique ID for this keypair
          const id = crypto.randomBytes(16).toString('hex');
          
          // Extract seed (first 32 bytes) and public key (last 32 bytes)
          const seed = keypairArray.slice(0, 32);
          const publicKey = keypairArray.slice(32, 64);
          
          // Publish to Redis pub/sub
          const message = JSON.stringify({
            type: 'keypair',
            id: id,
            keypair: keypairArray,
            timestamp: Date.now(),
            address: currentMatch.address || 'unknown'
          });

          await publisher.publish(PUBSUB_CHANNEL, message);
          console.log(`[Bridge] ✓ Published keypair ${id} (${currentMatch.address || 'unknown'}) to Redis`);
          
          // Reset match state after publishing
          currentMatch = {};
        } else {
          console.warn(`[Bridge] Invalid keypair array length: ${keypairArray.length}, expected 64`);
        }
      } catch (error) {
        // Only log if it looks like it might be a keypair array
        if (trimmed.length > 50) {
          console.error(`[Bridge] Error parsing keypair array: ${error}`);
        }
      }
    }
  });

  keygrinder.on('close', (code) => {
    console.log(`Keygrinder process exited with code ${code}`);
    publisher.quit();
    process.exit(code || 0);
  });

  keygrinder.on('error', (error) => {
    console.error(`Failed to start keygrinder: ${error}`);
    publisher.quit();
    process.exit(1);
  });

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nShutting down...');
    keygrinder.kill();
    await publisher.quit();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.log('\nShutting down...');
    keygrinder.kill();
    await publisher.quit();
    process.exit(0);
  });
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});

