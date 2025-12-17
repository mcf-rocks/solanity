#!/usr/bin/env ts-node

/**
 * CPU-based fallback keygrinder for testing on systems without CUDA/GPU
 * Outputs keys in the same format as the CUDA keygrinder
 */

import { Keypair } from '@solana/web3.js';

// Configuration (matching src/config.h)
const MAX_ITERATIONS = 100000;
const STOP_AFTER_KEYS_FOUND = 100;
const ATTEMPTS_PER_EXECUTION = 100000;

// Vanity patterns to search for - can be prefix (start) or suffix (end)
// For suffix matching, use negative length (e.g., -4 for "wewe" at end)
const PATTERNS: string[] = [
  'we',  // Match addresses ending with "we" (faster for testing - ~3 seconds vs ~30-60 minutes)
];

interface Config {
  maxIterations: number;
  stopAfterKeysFound: number;
  prefixes: string[];  // Note: despite the name, these are now suffix patterns
}

function getTimeStr(): string {
  const now = new Date();
  return now.toISOString().replace('T', ' ').substring(0, 19);
}

function generateRandomSeed(): Uint8Array {
  const seed = new Uint8Array(32);
  // Use crypto.randomBytes for secure random generation
  const randomBytes = require('crypto').randomBytes(32);
  for (let i = 0; i < 32; i++) {
    seed[i] = randomBytes[i];
  }
  return seed;
}

// Use Solana's library for keypair generation (handles ed25519 properly)
function generateKeypairFromSeed(seed: Uint8Array): { keypair: Keypair; address: string } {
  const keypair = Keypair.fromSeed(seed);
  const address = keypair.publicKey.toBase58();
  return { keypair, address };
}

function matchesPattern(address: string, patterns: string[]): string | null {
  for (const pattern of patterns) {
    let matches = true;
    
    // Check if pattern matches at the END of the address (suffix)
    if (address.length >= pattern.length) {
      const startPos = address.length - pattern.length;
      for (let i = 0; i < pattern.length; i++) {
        if (pattern[i] !== '?' && pattern[i] !== address[startPos + i]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        return pattern;
      }
    }
  }
  return null;
}

function seedToHex(seed: Uint8Array): string {
  return Array.from(seed)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

async function runVanitySearch(config: Config) {
  console.log('CPU Keygrinder Starting...');
  console.log(`Searching for addresses ending with: ${config.prefixes.join(', ')}`);
  console.log(`Max iterations: ${config.maxIterations}`);
  console.log(`Stop after ${config.stopAfterKeysFound} keys found`);
  console.log('');

  let keysFound = 0;
  let totalAttempts = 0;
  const startTime = Date.now();

  for (let iteration = 0; iteration < config.maxIterations; iteration++) {
    const iterationStart = Date.now();
    let iterationAttempts = 0;
    let iterationKeysFound = 0;

    // Generate and test keys
    for (let attempt = 0; attempt < ATTEMPTS_PER_EXECUTION; attempt++) {
      iterationAttempts++;
      totalAttempts++;

      // Generate random seed
      const seed = generateRandomSeed();
      
      // Generate keypair from seed
      const { keypair, address } = generateKeypairFromSeed(seed);

      // Check if it matches any pattern (suffix)
      const matchedPattern = matchesPattern(address, config.prefixes);
      
      if (matchedPattern) {
        iterationKeysFound++;
        keysFound++;

        // Output MATCH line (matching CUDA format: "GPU 0 MATCH <address> - <seed-hex>")
        console.log(`GPU 0 MATCH ${address} - ${seedToHex(seed)}`);

        // Output keypair array (matching CUDA format: [seed (32 bytes) + publicKey (32 bytes)])
        // Extract seed from keypair (first 32 bytes of secretKey) and public key (last 32 bytes)
        const secretKeyBytes = Array.from(keypair.secretKey);
        const seedBytes = secretKeyBytes.slice(0, 32);
        const publicKeyBytes = Array.from(keypair.publicKey.toBytes());
        const keypairArray = [...seedBytes, ...publicKeyBytes];
        console.log(JSON.stringify(keypairArray));

        if (keysFound >= config.stopAfterKeysFound) {
          console.log(`\nEnough keys found, Done!`);
          return;
        }
      }
    }

    const iterationTime = (Date.now() - iterationStart) / 1000;
    const attemptsPerSecond = iterationAttempts / iterationTime;

    console.log(
      `${getTimeStr()} Iteration ${iteration + 1} Attempts: ${iterationAttempts} in ${iterationTime.toFixed(3)} at ${attemptsPerSecond.toFixed(2)}cps - Total Attempts ${totalAttempts} - keys found ${keysFound}`
    );

    if (keysFound >= config.stopAfterKeysFound) {
      console.log(`\nEnough keys found, Done!`);
      return;
    }
  }

  console.log('\nIterations complete, Done!');
}

// Main
const config: Config = {
  maxIterations: MAX_ITERATIONS,
  stopAfterKeysFound: STOP_AFTER_KEYS_FOUND,
  prefixes: PATTERNS,
};

// Allow overriding config via environment variables
if (process.env.MAX_ITERATIONS) {
  config.maxIterations = parseInt(process.env.MAX_ITERATIONS, 10);
}
if (process.env.STOP_AFTER_KEYS_FOUND) {
  config.stopAfterKeysFound = parseInt(process.env.STOP_AFTER_KEYS_FOUND, 10);
}
if (process.env.PREFIXES) {
  config.prefixes = process.env.PREFIXES.split(',').map(p => p.trim());
}

runVanitySearch(config).catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});

