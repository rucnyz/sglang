"""Generate N synthetic LoRA adapters for a base model.

Each adapter has the same architecture (q/k/v/o + gate/up/down proj, rank R) but
random weights and a unique save dir / name. Used to drive the Sweep 2 LoRA
cache pressure benchmark — adapter content is irrelevant; what matters for
$V_{LoRA}(m_{LoRA})$ is per-adapter footprint and adapter count.

Usage:
  python dev/0/make_synthetic_loras.py \\
    --base-model Qwen/Qwen3-4B \\
    --out-dir /scratch/yuzhou/.cache/synthetic_loras/qwen3-4b-r16 \\
    --rank 16 --count 32

After running, the SGLang launch line is:
  --lora-paths lora_0=<dir>/lora_0 lora_1=<dir>/lora_1 ...
"""
import argparse, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--target-modules", nargs="+",
                    default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading base model {args.base_model} (CPU, BF16) to derive shapes...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    cfg = LoraConfig(
        r=args.rank, lora_alpha=args.rank*2,
        target_modules=args.target_modules,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )

    for i in range(args.count):
        # fresh peft model with random init each time
        torch.manual_seed(1000 + i)
        peft_model = get_peft_model(model, cfg, adapter_name=f"adapter_{i}")
        # randomize lora_A / lora_B (peft inits A as kaiming and B as zero;
        # nudge B with small random so it's a non-trivial adapter)
        with torch.no_grad():
            for n, p in peft_model.named_parameters():
                if "lora_B" in n:
                    p.normal_(mean=0.0, std=0.01)
        adapter_dir = out_dir / f"lora_{i}"
        peft_model.save_pretrained(str(adapter_dir), selected_adapters=[f"adapter_{i}"])
        # peft saves under <out>/adapter_i/{adapter_config.json,adapter_model.safetensors};
        # promote that subdir to be the adapter dir itself
        sub = adapter_dir / f"adapter_{i}"
        if sub.is_dir():
            for f in sub.iterdir():
                f.rename(adapter_dir / f.name)
            sub.rmdir()
        # detach this adapter so the next iteration starts fresh
        peft_model.delete_adapter(f"adapter_{i}")
        print(f"  wrote {adapter_dir}", flush=True)

    print(f"\nDone. {args.count} adapters @ rank {args.rank} under {out_dir}")
    print("Generate the launch flags with:")
    print(f"  --lora-paths $(for i in $(seq 0 {args.count-1}); do echo lora_$i={out_dir}/lora_$i; done)")

if __name__ == "__main__":
    main()
