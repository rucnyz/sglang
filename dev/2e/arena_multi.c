// Phase 2e.2.b — multi-pool C-side allocator.
//
// Each pool has its own bump allocator over a Python-supplied VA sub-range.
// PyTorch's CUDAPluggableAllocator dlopens this .so and looks up
// `pool0_malloc`/`pool0_free` for the first MemPool, `pool1_malloc`/`pool1_free`
// for the second. Each pair indexes into its own slot of `pools[]`.
//
// Capacity is mutable: Python calls `multi_set_capacity(pid, n)` after
// `arena.grow(pool, k)` to widen the bump-cap for that pool.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    uintptr_t base;
    size_t chunk_size;
    size_t n_chunks;  // mutable: how many chunks are currently mapped
    size_t next;
} pool_state_t;

#define N_POOLS 4
static pool_state_t pools[N_POOLS];

void multi_init(int pid, uintptr_t base, size_t chunk_size, size_t n_chunks) {
    if (pid < 0 || pid >= N_POOLS) {
        fprintf(stderr, "[multi] init: invalid pid %d\n", pid);
        return;
    }
    pools[pid].base = base;
    pools[pid].chunk_size = chunk_size;
    pools[pid].n_chunks = n_chunks;
    pools[pid].next = 0;
    fprintf(stderr,
        "[multi] init pid=%d base=0x%lx chunk=%zu n=%zu\n",
        pid, (unsigned long) base, chunk_size, n_chunks);
}

void multi_set_capacity(int pid, size_t n_chunks) {
    if (pid < 0 || pid >= N_POOLS) return;
    pools[pid].n_chunks = n_chunks;
    fprintf(stderr, "[multi] pid=%d capacity := %zu\n", pid, n_chunks);
}

static void* alloc_from(int pid, ptrdiff_t size) {
    pool_state_t* p = &pools[pid];
    if ((size_t) size > p->chunk_size) {
        fprintf(stderr, "[multi pid=%d] alloc size=%td > chunk=%zu, refusing\n",
                pid, size, p->chunk_size);
        return NULL;
    }
    if (p->next >= p->n_chunks) {
        fprintf(stderr, "[multi pid=%d] alloc size=%td but capacity exhausted (next=%zu n=%zu)\n",
                pid, size, p->next, p->n_chunks);
        return NULL;
    }
    void* va = (void*) (p->base + p->next * p->chunk_size);
    fprintf(stderr, "[multi pid=%d] alloc size=%td -> chunk %zu va=0x%lx\n",
            pid, size, p->next, (unsigned long) (uintptr_t) va);
    p->next += 1;
    return va;
}

#define POOL_MALLOC(N) \
void* pool##N##_malloc(ptrdiff_t size, int device, void* stream) { \
    (void) device; (void) stream; \
    return alloc_from(N, size); \
} \
void pool##N##_free(void* p, ptrdiff_t s, int d, void* st) { \
    (void) p; (void) s; (void) d; (void) st; \
}

POOL_MALLOC(0)
POOL_MALLOC(1)
POOL_MALLOC(2)
POOL_MALLOC(3)
