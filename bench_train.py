import sys, runpy
# scripts/train_stage1.sh points at train/run.py, an outdated path; the real entry is `python -m tabicl.train`.
# Scaled down so one card can run it: stage 1 is originally max_steps=100000 / batch 512.
if __name__ == "__main__":
    sys.argv = ["tabicl.train",
                "--device", "cuda", "--dtype", "bfloat16",
                "--max_steps", "50", "--batch_size", "32", "--micro_batch_size", "4",
                "--lr", "1e-4", "--scheduler", "cosine_warmup", "--warmup_proportion", "0.02",
                "--gradient_clipping", "1.0",
                "--prior_type", "mix_scm", "--prior_device", "cpu", "--batch_size_per_gp", "4",
                "--np_seed", "42", "--torch_seed", "42", "--wandb_log", "False"]
    runpy.run_module("tabicl.train", run_name="__main__")
