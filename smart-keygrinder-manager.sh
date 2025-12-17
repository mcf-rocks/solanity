#!/bin/bash

# Smart Keygrinder Manager
# Only starts keygrinder if Redis cache drops below threshold
# Auto-restarts on spot instance interruption

set -e

# Configuration
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_URL="${REDIS_URL:-redis://${REDIS_HOST}:${REDIS_PORT}}"
CACHE_THRESHOLD="${CACHE_THRESHOLD:-80}"  # Start keygrinder if cache < 80
CACHE_MIN="${CACHE_MIN:-50}"  # Minimum cache before starting (safety check)
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"  # Check cache every 30 seconds
KEYGRINDER_DIR="${KEYGRINDER_DIR:-$(cd "$(dirname "$0")" && pwd)}"
LOG_FILE="${LOG_FILE:-/var/log/keygrinder-manager.log}"

# Colors for logging
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${GREEN}INFO${NC}: $1"
}

log_warn() {
    log "${YELLOW}WARN${NC}: $1"
}

log_error() {
    log "${RED}ERROR${NC}: $1"
}

# Check Redis cache count
get_cache_count() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ZCARD keypair:list 2>/dev/null || echo "0"
}

# Check if Redis is accessible
check_redis() {
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING >/dev/null 2>&1
}

# Check if keygrinder is running
is_keygrinder_running() {
    pgrep -f "keygrinder-bridge\|cuda_ed25519_vanity" >/dev/null 2>&1
}

# Start keygrinder
start_keygrinder() {
    log_info "Starting keygrinder..."
    cd "$KEYGRINDER_DIR"
    
    # Start bridge in background, redirect output to log
    nohup npm run bridge >> "$LOG_FILE" 2>&1 &
    
    # Wait a moment to see if it starts successfully
    sleep 5
    
    if is_keygrinder_running; then
        log_info "Keygrinder started successfully (PID: $(pgrep -f 'keygrinder-bridge'))"
        return 0
    else
        log_error "Failed to start keygrinder"
        return 1
    fi
}

# Stop keygrinder gracefully
stop_keygrinder() {
    if is_keygrinder_running; then
        log_info "Stopping keygrinder..."
        pkill -TERM -f "keygrinder-bridge\|cuda_ed25519_vanity" || true
        sleep 2
        
        # Force kill if still running
        if is_keygrinder_running; then
            log_warn "Force killing keygrinder..."
            pkill -KILL -f "keygrinder-bridge\|cuda_ed25519_vanity" || true
        fi
        
        log_info "Keygrinder stopped"
    fi
}

# Main monitoring loop
monitor_loop() {
    log_info "Starting smart keygrinder manager"
    log_info "Configuration:"
    log_info "  Redis: ${REDIS_HOST}:${REDIS_PORT}"
    log_info "  Cache threshold: ${CACHE_THRESHOLD}"
    log_info "  Cache minimum: ${CACHE_MIN}"
    log_info "  Check interval: ${CHECK_INTERVAL}s"
    
    while true; do
        # Check Redis connectivity
        if ! check_redis; then
            log_error "Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
            log_warn "Will retry in ${CHECK_INTERVAL} seconds..."
            sleep "$CHECK_INTERVAL"
            continue
        fi
        
        # Get current cache count
        cache_count=$(get_cache_count)
        log_info "Current cache count: ${cache_count}"
        
        # Check if keygrinder is running
        keygrinder_running=$(is_keygrinder_running && echo "yes" || echo "no")
        
        # Decision logic
        if [ "$cache_count" -lt "$CACHE_MIN" ]; then
            # Cache is critically low - start keygrinder if not running
            if [ "$keygrinder_running" = "no" ]; then
                log_warn "Cache critically low (${cache_count} < ${CACHE_MIN}), starting keygrinder..."
                start_keygrinder
            else
                log_info "Cache low but keygrinder already running"
            fi
        elif [ "$cache_count" -lt "$CACHE_THRESHOLD" ]; then
            # Cache below threshold - start keygrinder if not running
            if [ "$keygrinder_running" = "no" ]; then
                log_info "Cache below threshold (${cache_count} < ${CACHE_THRESHOLD}), starting keygrinder..."
                start_keygrinder
            else
                log_info "Cache below threshold but keygrinder already running"
            fi
        else
            # Cache is healthy - stop keygrinder if running (save resources)
            if [ "$keygrinder_running" = "yes" ]; then
                log_info "Cache healthy (${cache_count} >= ${CACHE_THRESHOLD}), stopping keygrinder to save resources..."
                stop_keygrinder
            else
                log_info "Cache healthy, keygrinder not needed"
            fi
        fi
        
        # Wait before next check
        sleep "$CHECK_INTERVAL"
    done
}

# Handle signals for graceful shutdown
cleanup() {
    log_info "Shutting down..."
    stop_keygrinder
    exit 0
}

trap cleanup SIGTERM SIGINT

# Check if running as systemd service or directly
if [ -n "$1" ]; then
    case "$1" in
        start)
            monitor_loop
            ;;
        stop)
            stop_keygrinder
            ;;
        status)
            cache_count=$(get_cache_count)
            keygrinder_running=$(is_keygrinder_running && echo "running" || echo "stopped")
            echo "Cache count: ${cache_count}"
            echo "Keygrinder: ${keygrinder_running}"
            ;;
        *)
            echo "Usage: $0 {start|stop|status}"
            exit 1
            ;;
    esac
else
    # Run directly if no arguments
    monitor_loop
fi

