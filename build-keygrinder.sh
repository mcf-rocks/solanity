#!/bin/bash

# Helper script to build the keygrinder

set -e

echo "=========================================="
echo "Building Solanity Keygrinder"
echo "=========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA compiler (nvcc) not found in PATH"
    echo ""
    echo "Please install CUDA Toolkit and add it to your PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    echo ""
    echo "Common CUDA installation locations:"
    echo "  - /usr/local/cuda/bin/nvcc"
    echo "  - /opt/cuda/bin/nvcc"
    echo ""
    echo "Or if using conda:"
    echo "  conda install -c nvidia cuda-toolkit"
    exit 1
fi

echo "✓ Found CUDA compiler: $(which nvcc)"
echo "  Version: $(nvcc --version | grep release | head -1)"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ Found NVIDIA GPU:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo ""
else
    echo "⚠ Warning: nvidia-smi not found"
    echo "  You may not have NVIDIA GPU drivers installed"
    echo "  The build may still work, but you won't be able to run the keygrinder"
    echo ""
fi

# Build
echo "Building keygrinder..."
echo ""

export PATH=/usr/local/cuda/bin:$PATH

# Determine number of cores
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
elif [ "$(uname)" = "Darwin" ]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=4
fi

echo "Using $CORES parallel jobs"
make -j$CORES

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Keygrinder executable: ./src/release/cuda_ed25519_vanity"
echo ""

if [ -f "./src/release/cuda_ed25519_vanity" ]; then
    echo "✓ Build successful!"
    ls -lh ./src/release/cuda_ed25519_vanity
    echo ""
    echo "You can now run: npm run bridge"
else
    echo "❌ Build failed - executable not found"
    exit 1
fi

