run_K_kv_off_matrix_20260526_234639_cycle1_baseline: 5573 requests parsed
run_K_full_matrix_20260526_234639_cycle2_ours: 5435 requests parsed
run_K_kv_off_matrix_20260526_234639_cycle3_baseline: 5452 requests parsed
run_K_full_matrix_20260526_234639_cycle4_ours: 5618 requests parsed
run_K_kv_off_matrix_20260526_234639_cycle5_baseline: 5306 requests parsed
run_K_full_matrix_20260526_234639_cycle6_ours: 5442 requests parsed
# Per-request TTFT/queue analysis across N=3 matrix

## baseline (N=16331 requests across 3 cycles)

* e2e_latency (s):        n=16331 mean=8.692 std=97.574 p50=0.498 p90=4.279 p99=84.568 max=2122.305
* queue_time (s):         n=16331 mean=0.000 std=0.000 p50=0.000 p90=0.000 p99=0.000 max=0.000
* prompt_tokens:          n=16331 mean=7296.207 std=3846.163 p50=6885.000 p90=11007.000 p99=21945.000 max=25171.000
* cached_tokens:          n=16331 mean=7112.524 std=3856.909 p50=6656.000 p90=10752.000 p99=21760.000 max=25088.000
* completion_tokens:      n=16331 mean=270.871 std=3339.063 p50=1.000 p90=104.000 p99=2625.000 max=63963.000
* hit_ratio:              n=16331 mean=0.965 std=0.062 p50=0.976 p90=0.991 p99=0.996 max=0.999
* e2e_latency cached>0:   n=16325 mean=8.632 std=97.432 p50=0.498 p90=4.262 p99=82.294 max=2122.305
* e2e_latency cached=0:   n=6 mean=172.531 std=264.055 p50=6.366 p90=593.389 p99=593.389 max=593.389
* api_dispatch_lag (ms):  n=16331 mean=26.965 std=17.657 p50=25.280 p90=43.086 p99=71.001 max=752.151
* post_send_lag (ms):     n=16331 mean=0.559 std=1.423 p50=0.335 p90=1.207 p99=2.664 max=64.664
* Σ e2e_latency = 141956 s; per-trial-equivalent = **1478.7 s**

## ours (N=16495 requests across 3 cycles)

* e2e_latency (s):        n=16495 mean=8.315 std=96.297 p50=0.522 p90=4.433 p99=48.707 max=2132.017
* queue_time (s):         n=16495 mean=0.000 std=0.000 p50=0.000 p90=0.000 p99=0.000 max=0.000
* prompt_tokens:          n=16495 mean=6638.427 std=3050.890 p50=6524.000 p90=9946.000 p99=17464.000 max=25600.000
* cached_tokens:          n=16495 mean=6459.705 std=3064.139 p50=6400.000 p90=9728.000 p99=17152.000 max=25344.000
* completion_tokens:      n=16495 mean=248.975 std=3302.747 p50=1.000 p90=102.000 p99=1063.000 max=64068.000
* hit_ratio:              n=16495 mean=0.964 std=0.059 p50=0.974 p90=0.990 p99=0.995 max=0.998
* e2e_latency cached>0:   n=16488 mean=8.316 std=96.317 p50=0.521 p90=4.416 p99=48.707 max=2132.017
* e2e_latency cached=0:   n=7 mean=7.082 std=6.668 p50=5.657 p90=21.799 p99=21.799 max=21.799
* api_dispatch_lag (ms):  n=16495 mean=27.155 std=18.101 p50=25.898 p90=43.096 p99=63.494 max=939.938
* post_send_lag (ms):     n=16495 mean=0.585 std=1.530 p50=0.327 p90=1.265 p99=2.950 max=56.556
* Σ e2e_latency = 137164 s; per-trial-equivalent = **1428.8 s**

## Side-by-side: baseline (16331 reqs) vs ours (16495 reqs)

* e2e_latency (s): baseline mean=8.692, ours mean=8.315, Δ=-0.377
* queue_time (s): baseline mean=0.000, ours mean=0.000, Δ=+0.000
* cached_tokens: baseline mean=7112.524, ours mean=6459.705, Δ=-652.819
* prompt_tokens: baseline mean=7296.207, ours mean=6638.427, Δ=-657.780
* completion_tokens: baseline mean=270.871, ours mean=248.975, Δ=-21.896
