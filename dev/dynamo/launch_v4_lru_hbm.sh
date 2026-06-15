#!/usr/bin/env bash
# HBM-ONLY LRU baseline: no aginfer scorer (SGLANG_KV_POLICY_MODULE unset) + pure
# --radix-eviction-policy lru. The do-no-harm floor. Identical to fork_hbm EXCEPT the
# eviction policy + scorer — the only lever under test.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/workspace/sglang/python:/workspace/sglang/dev/aginfer:${PYTHONPATH:-}
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export DYN_SYSTEM_PORT=8081
exec python -m dynamo.sglang --model deepseek-ai/DeepSeek-V4-Flash --served-model-name __backend_v4 --tp-size 2 --trust-remote-code \
  --page-size 64 --moe-runner-backend flashinfer_mxfp4 --disable-flashinfer-autotune \
  --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.1 --mem-fraction-static 0.83 --context-length 65536 \
  --reasoning-parser deepseek-r1 \
  --enable-rl --aginfer-notify-url http://127.0.0.1:9100 --aginfer-theta-hi 0.85 --aginfer-theta-lo 0.70 --aginfer-theta-crit 0.90 --aginfer-heartbeat-s 5.0 \
  --radix-eviction-policy lru --max-total-tokens 131072 \
  --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:20080\",\"enable_kv_cache_events\":true}"
