// Phase 2e.1.b — minimal CUDA arena allocator behind torch.cuda.MemPool.
//
// PyTorch's CUDAPluggableAllocator dlopens this .so and looks up
// `arena_malloc` and `arena_free` by name. Our implementation:
//
//   - Holds a Python-supplied (VA base, chunk_size, n_chunks) arena descriptor.
//   - All chunks are pre-mapped (cuMemMap) by Python before the first
//     allocation hits us. We just hand them out in order.
//   - This is a dumb bump allocator. arena_free is a no-op; the test never
//     reuses chunks. Phase 2e.2 will replace this with a chunk-bitmap allocator.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uintptr_t g_arena_base = 0;
static size_t g_arena_chunk_size = 0;
static size_t g_arena_n_chunks = 0;
static size_t g_arena_next = 0;

// Called from Python (via ctypes) before the allocator is used.
void arena_init(uintptr_t base, size_t chunk_size, size_t n_chunks) {
    g_arena_base = base;
    g_arena_chunk_size = chunk_size;
    g_arena_n_chunks = n_chunks;
    g_arena_next = 0;
    fprintf(stderr,
        "[arena] init base=0x%lx chunk=%zu n=%zu\n",
        (unsigned long) base, chunk_size, n_chunks);
}

// Hand out the next chunk. Size must fit; otherwise NULL.
void* arena_malloc(ptrdiff_t size, int device, void* stream) {
    (void) device;
    (void) stream;
    if ((size_t) size > g_arena_chunk_size) {
        fprintf(stderr,
            "[arena] alloc size=%td > chunk=%zu, refusing\n",
            size, g_arena_chunk_size);
        return NULL;
    }
    if (g_arena_next >= g_arena_n_chunks) {
        fprintf(stderr,
            "[arena] alloc size=%td but arena exhausted (next=%zu n=%zu)\n",
            size, g_arena_next, g_arena_n_chunks);
        return NULL;
    }
    void* p = (void*) (g_arena_base + g_arena_next * g_arena_chunk_size);
    fprintf(stderr,
        "[arena] alloc size=%td -> chunk %zu va=0x%lx\n",
        size, g_arena_next, (unsigned long) (uintptr_t) p);
    g_arena_next += 1;
    return p;
}

void arena_free(void* ptr, ptrdiff_t size, int device, void* stream) {
    (void) ptr; (void) size; (void) device; (void) stream;
    // No-op. Smoke test does not reuse chunks.
}
