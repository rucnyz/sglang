// 64-slot multi-pool C allocator.
//
// Same mechanism as arena_multi.c but with 64 pool pairs, enough to back
// every layer-tensor of a typical KV pool (e.g., Qwen3-Next has ~30 attn
// layers × 2 = 60 sub-pools). PyTorch's CUDAPluggableAllocator dlopens
// this and looks up `pool##N##_malloc` / `pool##N##_free` per MemPool.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uintptr_t base;
    size_t chunk_size;
    size_t n_chunks;  // mutable
    size_t next;
} pool_state_t;

#define N_POOLS 64
static pool_state_t pools[N_POOLS];

void multi_init(int pid, uintptr_t base, size_t chunk_size, size_t n_chunks) {
    if (pid < 0 || pid >= N_POOLS) {
        fprintf(stderr, "[multi64] init: invalid pid %d\n", pid);
        return;
    }
    pools[pid].base = base;
    pools[pid].chunk_size = chunk_size;
    pools[pid].n_chunks = n_chunks;
    pools[pid].next = 0;
}

void multi_set_capacity(int pid, size_t n_chunks) {
    if (pid < 0 || pid >= N_POOLS) return;
    pools[pid].n_chunks = n_chunks;
}

static void* alloc_from(int pid, ptrdiff_t size) {
    pool_state_t* p = &pools[pid];
    // How many chunks does this allocation span? Round up.
    size_t n_needed = ((size_t) size + p->chunk_size - 1) / p->chunk_size;
    if (p->next + n_needed > p->n_chunks) return NULL;
    void* va = (void*) (p->base + p->next * p->chunk_size);
    p->next += n_needed;
    return va;
}

#define POOL(N) \
void* pool##N##_malloc(ptrdiff_t s, int d, void* st) { \
    (void) d; (void) st; return alloc_from(N, s); \
} \
void pool##N##_free(void* p, ptrdiff_t s, int d, void* st) { \
    (void) p; (void) s; (void) d; (void) st; \
}

POOL(0)  POOL(1)  POOL(2)  POOL(3)  POOL(4)  POOL(5)  POOL(6)  POOL(7)
POOL(8)  POOL(9)  POOL(10) POOL(11) POOL(12) POOL(13) POOL(14) POOL(15)
POOL(16) POOL(17) POOL(18) POOL(19) POOL(20) POOL(21) POOL(22) POOL(23)
POOL(24) POOL(25) POOL(26) POOL(27) POOL(28) POOL(29) POOL(30) POOL(31)
POOL(32) POOL(33) POOL(34) POOL(35) POOL(36) POOL(37) POOL(38) POOL(39)
POOL(40) POOL(41) POOL(42) POOL(43) POOL(44) POOL(45) POOL(46) POOL(47)
POOL(48) POOL(49) POOL(50) POOL(51) POOL(52) POOL(53) POOL(54) POOL(55)
POOL(56) POOL(57) POOL(58) POOL(59) POOL(60) POOL(61) POOL(62) POOL(63)
