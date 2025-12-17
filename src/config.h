#ifndef VANITY_CONFIG
#define VANITY_CONFIG

static int const MAX_ITERATIONS = 100;  // Safety limit: ~2 hours max (stops early at 100 keys)
static int const STOP_AFTER_KEYS_FOUND = 100;

// how many times a gpu thread generates a public key in one go
__device__ const int ATTEMPTS_PER_EXECUTION = 100000;

__device__ const int MAX_PATTERNS = 10;

// exact matches - can be prefix (beginning) or suffix (end) of address, letter ? is wildcard
// For suffix matching, use negative pattern length (e.g., -4 for "WEWE" at end)

__device__ static char const *prefixes[] = {
	"wewe",  // Match addresses ending with "wewe" (lowercase)
};


#endif
