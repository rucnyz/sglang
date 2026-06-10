"""Offline tier-link characterization — the groundwork that should precede S1.

Measures the actual per-transfer latency / effective bandwidth of the links a
demoted prefix's KV traverses on THIS box, at realistic KV sizes (a 12K-token
full-KV ≈ 14 MB; 100K ≈ 117 MB). Tells us where load_back is cheap vs expensive
-> where the S1 predictive-promote can actually win.

Links:
  HBM<-DRAM  (load_back / predictive promote)   : host-pinned -> GPU  (H2D)
  HBM->DRAM  (write_through / demote)            : GPU -> host-pinned  (D2H)
  DISK->host (mooncake/SSD read)                 : file read
  DISK->HBM  (cold load_back from disk)          : file read + H2D
Each at sizes spanning the realistic KV range; warm + median of N.
"""
import torch, time, os, tempfile, statistics

# Portable scratch dir for the DISK-tier read benchmark (override with LINKCHAR_TMPDIR).
_DISK_DIR = os.environ.get("LINKCHAR_TMPDIR", "/tmp/linkchar")
os.makedirs(_DISK_DIR, exist_ok=True)

KV_PER_TOKEN = 1.17e3  # bytes/token (measured: 4.19MB / 3584 tok)
SIZES = {  # tokens -> bytes
    "12K": int(12000 * KV_PER_TOKEN),
    "24K": int(24000 * KV_PER_TOKEN),
    "50K": int(50000 * KV_PER_TOKEN),
    "100K": int(100000 * KV_PER_TOKEN),
}
N = 7
dev = "cuda:0"


def med_ms(fn):
    fn()  # warm
    ts = []
    for _ in range(N):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


print(f"{'size':>6} {'MB':>7} | {'DRAM->HBM(H2D)':>16} {'HBM->DRAM(D2H)':>16} {'DISK->host':>12} {'DISK->HBM':>12}")
for name, nbytes in SIZES.items():
    n_el = nbytes // 4
    host = torch.empty(n_el, dtype=torch.float32, pin_memory=True)
    gpu = torch.empty(n_el, dtype=torch.float32, device=dev)
    host.fill_(1.0)
    h2d = med_ms(lambda: gpu.copy_(host, non_blocking=True))
    d2h = med_ms(lambda: host.copy_(gpu, non_blocking=True))
    # disk
    with tempfile.NamedTemporaryFile(delete=False, dir=_DISK_DIR) as f:
        path = f.name
        f.write(host.numpy().tobytes())
    os.system("sync")  # flush; then drop-cache read not available without root -> approximate warm
    def disk_read():
        with open(path, "rb", buffering=0) as fh:
            b = fh.read()
        return b
    # DISK->host (read into a host buffer)
    def disk_to_host():
        with open(path, "rb", buffering=0) as fh:
            buf = fh.read()
    dh = med_ms(disk_to_host)
    def disk_to_hbm():
        with open(path, "rb", buffering=0) as fh:
            buf = fh.read()
        t = torch.frombuffer(bytearray(buf), dtype=torch.float32)
        gpu.copy_(t.pin_memory(), non_blocking=True)
    dhbm = med_ms(disk_to_hbm)
    os.unlink(path)
    mb = nbytes / 1e6
    def bw(ms): return mb / (ms / 1000) / 1000  # GB/s
    print(f"{name:>6} {mb:>7.1f} | {h2d:>7.2f}ms {bw(h2d):>4.0f}GB/s {d2h:>7.2f}ms {bw(d2h):>4.0f}GB/s "
          f"{dh:>6.1f}ms {dhbm:>6.1f}ms")
print("\nload_back = the cost S1's predictive promote moves OFF the resume critical path.")
print("DRAM->HBM cheap => single-resume TTFT win small; DISK->HBM (and aggregate) is where S1 pays.")
