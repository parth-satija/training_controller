"""Sample associate script for training_controller (an "evaluation" run).

Associate scripts go with one training script: evaluation, inference, data
checks. They are registered while that training script is active and have their
own configs. Build the Rust module once with `cargo build --features python`,
then in the REPL:

    /add_scripts sample D:\\training_controller\\python_sample.py
    /open_script sample
    /run_script quick                     # produce a few checkpoints first

    /add_associate_script eval D:\\training_controller\\associate_sample.py
    /open_associate_script eval
    /add_associate_config quick --split val --samples 3 --delay 0.2
    /add_associate_config broken --split val --samples 3 --fail
    /show_associate_configs
    /select_associate_config quick
    /test                                 # runs eval with the "quick" config
    /test --samples 8                     # extra flags are added after the config's
    /select_associate_config broken
    /test                                 # exits with an [error] and code 1
    /select_associate_config              # no name: run without a config

Watch it with /panel output, /panel error, /panel log or /show_active_panels.

Associate scripts belong to the training script that was active when they were
added (here: sample). /show_associate_scripts lists that script's associate
scripts, and selecting another training script clears the associate selection.
/test points tc.latest_checkpoint() at the training script's checkpoint folder,
so this script evaluates what that script stored.
"""

import argparse
import pickle
import random
import sys
import time

import training_controller as tc


def parse_args():
    parser = argparse.ArgumentParser(description="training_controller associate sample")
    parser.add_argument("--split", default="val", help="data split to evaluate")
    parser.add_argument("--samples", type=int, default=5, help="number of samples")
    parser.add_argument("--delay", type=float, default=0.3, help="seconds per sample")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--fail", action="store_true", help="exit with an error after the first sample"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("Received flags:")
    for key, value in vars(args).items():
        print(f"  {key} = {value!r}")
    print(f"[log] evaluating split '{args.split}' on {args.samples} samples")

    path = tc.latest_checkpoint()
    if path is None:
        print("[error] no checkpoint found; run a training script first (/run_script)")
        sys.exit(1)
    print(f"[checkpoint] evaluating {path}")

    # The sample training script stores a pickled dict (see python_sample.py).
    with open(path, "rb") as file:
        state = pickle.load(file)
    base_loss = float(state.get("loss", 1.0))
    print(f"[log] checkpoint from epoch {state.get('epoch')} with training loss {base_loss:.4f}")

    total = 0.0
    for sample in range(1, args.samples + 1):
        loss = base_loss * 1.1 + random.random() * 0.02
        total += loss
        print(f"[metrics] sample={sample} loss={loss:.4f}")
        if args.fail:
            print(f"[error] simulated failure at sample {sample}")
            sys.exit(1)
        time.sleep(args.delay)

    print(f"[log] mean loss on '{args.split}': {total / max(args.samples, 1):.4f}")


if __name__ == "__main__":
    main()
