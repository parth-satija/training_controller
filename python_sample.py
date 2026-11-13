"""Sample script for training_controller: output panels, flag configs, Rust module,
the live training dashboard.

The default run takes about 10 seconds, so you can open panels while it runs.
Build the Rust module once with `cargo build --features python`, then in the
training_controller REPL:

    /add_scripts sample D:\\training_controller\\python_sample.py
    /open_script sample
    /run_script                  # starts in the background and returns at once

Watch it live (q / Esc / Ctrl+C closes a view and returns to the REPL):

    /dashboard training          # the training dashboard (see below)
    /show_active_panels          # panels, line counts, and run status
    /panel output                # everything printed, exactly as printed
    /panel error                 # only "[error] ..." lines, in red
    /panel log                   # only "[log] ..." lines
    /panel metrics               # "[metrics] ..." -> a panel named "metrics"
    /panel text_example          # "[text_example] ..." -> its own panel
    /panel training_controller   # output of the Rust module (tc.test)
    /panel other                 # prints without a [tag] prefix
    /clear_panels                # forget the captured output

Combine it with flag configs:

    /add_config quick --epochs 4 --delay 0.5
    /add_config noisy --simulate-errors --name "noisy run"
    /add_config risky --lr 0.9           # a high learning rate also logs an [error]
    /show_configs
    /run_script noisy                    # watch /panel error fill up in red
    /run_script quick --seed 7           # saved flags plus an extra flag

TRAINING DASHBOARD
------------------
The script declares what to show with three calls (see main() below):

    tc.init_training_dashboard("loss", "curve", "acc", "graph", "lr", "box")
    tc.update_training_dashboard(loss, acc, lr)     # one value per variable
    tc.update_hero_text("Training at Epoch 47, step 4700, Total Loss at 0.0023")

These calls only record events to training_dashboard.jsonl next to this file
(/run_script tells Python where). They never open a UI by themselves: run
`/dashboard training` in the REPL, while the script runs or after it has
finished. It shows every variable, the bold hero text in the center, and the
time spent training (the clock stops when the script ends). Try it without
Python at all via `/dashboard training --demo`.

Widget types: "curve" = line plot of the whole history, "graph" = bar chart of
the latest values, "box" = a block with the latest value, its change, min/max.
Colors, labels, decimals and the hero style come from the dashboard config file
(`/dashboard_config`).

The script also writes metrics.jsonl next to itself, so `/dashboard` can
display the run.

It also stores its metrics in the controller's SQLite log (tc.init_log /
tc.log). Every /run_script gets an experiment ID, so afterwards:

    /show_experiments            # all runs: ID, status, steps, log columns
    /show_log 1 --last 5         # the logged values of experiment 1
    /graph 1                     # every column against time, one graph each
    /graph 1 loss + accuracy     # two graphs at once (x defaults to time)
    /graph 1,2 loss step         # loss vs step, experiments 1 and 2 overlaid
    /graph 1 loss + accuracy --save curves   # save these graphs ...
    /graph curves 2              # ... and show them for experiment 2

CHECKPOINTS
-----------
Every epoch the script calls tc.save_checkpoint(data, is_best, step=...,
metrics=...). `data` is any bytes (here a pickled dict; with PyTorch use
torch.save into an io.BytesIO). Nothing is deleted by default; you choose what
is kept, per script, in the REPL:

    /checkpoint_preset                       # list the presets
    /checkpoint_preset last_and_best         # keep the last 2 + the best of each run
    /set_checkpoint_config keep_last 3       # ...or fine-tune: keep the last 3
    /set_checkpoint_config keep_every 5      # ...plus every 5th epoch
    /set_checkpoint_config max_total_mb 500  # ...and cap the folder at 500 MB
    /show_checkpoint_config                  # all settings and their values
    /run_script quick                        # watch /panel checkpoint
    /checkpoints                             # ID, run, step, best/pin flags, size, metrics
    /pin_checkpoint best                     # protect it from pruning forever
    /prune_checkpoints dry                   # preview what the settings would delete
    /prune_checkpoints                       # ...and delete it
    /verify_checkpoints                      # re-check every file's SHA-256
    /export_checkpoint latest D:\\exports      # copy one out (never overwrites)
    /remove_checkpoint 3                     # delete one (asks first)

Retention rules apply per run, so a new run never deletes an older run's
checkpoints. A later run can resume with tc.latest_checkpoint().

PAUSE AND RESUME
----------------
The script registers a pause method with tc.register_pause(pause) (see main()
below). While it runs, in the REPL:

    /pause       # the controller calls pause() inside the running script
    /panel pause # follow what the pause method does ([pause] lines)
    /resume      # run the script again: same flags as the last /run_script
                 # (kept in tc.json, even after restarting the controller)
                 # plus --resume

pause() runs on a background thread, so here it only raises a flag. The training
loop checks it after every epoch (right after that epoch's checkpoint was
stored) and returns, so the script exits with code 0 and the run is recorded as
"paused" (/show_experiments). With --resume the script loads
tc.latest_checkpoint() and continues at the next epoch. Every resume is a new
experiment, and it restarts the training dashboard; overlay the parts of a run
with e.g. `/graph 1,2 loss step`.
"""

import argparse
import json
import math
import pickle
import random
import threading
import time
from pathlib import Path

import training_controller as tc


def parse_args():
    parser = argparse.ArgumentParser(description="training_controller sample script")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--name", default="default", help="label for this run")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds per epoch")
    parser.add_argument("--verbose", action="store_true", help="print extra detail")
    parser.add_argument(
        "--simulate-errors",
        action="store_true",
        help="emit an [error] line every third epoch",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint (added by the controller's /resume)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # `/pause` in the controller calls this function (on a background thread).
    # It only raises a flag; the training loop below checks it after every
    # epoch, so that epoch's checkpoint is stored before the script exits.
    pause_requested = threading.Event()

    def pause():
        print("[log] pause method called")
        pause_requested.set()

    if not tc.register_pause(pause):
        print("[log] not started by the controller: /pause cannot reach this script")

    # Routing: a line starting with "[tag]" goes to the panel called "tag"
    # ("[error]" is shown in red); a line without a tag goes to "other".
    # Every line also appears, as printed, in the "output" panel.

    # No tag -> "other" panel.
    print("Received flags:")
    for key, value in vars(args).items():
        print(f"  {key} = {value!r}")

    print(f"[log] starting run '{args.name}'")
    print("[text_example] panels are created the first time a tag is used")
    if args.lr > 0.5:
        print(f"[error] learning rate {args.lr} is very high; training may diverge")

    # The Rust module prints "[training_controller] ...", so its output gets
    # its own panel as well.
    tc.test(f"Starting run '{args.name}' ({args.epochs} epochs, lr={args.lr})")

    metrics_path = Path(__file__).with_name("metrics.jsonl")

    # Declare the log columns once; the controller owns the SQLite database and
    # returns the experiment ID it assigned to this run.
    experiment_id = tc.init_log("loss", "accuracy", "lr")
    print(f"[log] experiment #{experiment_id}")

    # Checkpoints of earlier runs are still there. A fresh run ignores them;
    # `/resume` starts this script with --resume, and then it continues from
    # the newest one (see "PAUSE AND RESUME" at the top).
    best_loss = float("inf")
    loss = float("nan")
    start_epoch = 1
    previous = tc.latest_checkpoint()  # a path, or None
    if args.resume:
        if previous:
            saved = pickle.loads(Path(previous).read_bytes())
            start_epoch = saved["epoch"] + 1
            loss = saved["loss"]
            best_loss = saved.get("best_loss", loss)
            print(f"[log] resuming from {previous}: continuing at epoch {start_epoch}")
        else:
            print("[log] --resume was given but there is no checkpoint; starting at epoch 1")
    elif previous:
        print(f"[checkpoint] newest stored checkpoint (could resume from it): {previous}")
    epoch = start_epoch - 1  # the last finished epoch

    # Declare the dashboard layout once, as alternating (name, type) pairs.
    # The order here is the order of the values passed to
    # tc.update_training_dashboard() below. Calling this again starts a fresh
    # run (open dashboards reset). It does NOT open the dashboard; that is
    # `/dashboard training` in the REPL.
    tc.init_training_dashboard(
        "loss", "curve",         # line plot of the whole history
        "val_loss", "curve",     # a second line plot
        "accuracy", "graph",     # bar chart of the latest values
        "lr", "box",             # block with the latest value and its change
    )
    tc.update_hero_text(f"Starting run '{args.name}'...")  # bold, centered text

    # A resumed run appends to the metrics file instead of overwriting it.
    with open(metrics_path, "a" if args.resume else "w", encoding="utf-8") as metrics:
        for epoch in range(start_epoch, args.epochs + 1):
            # Fake training curve: higher lr converges faster.
            loss = 2.0 * math.exp(-epoch * args.lr * 20) + random.random() * 0.05
            val_loss = loss * 1.1 + random.random() * 0.02
            accuracy = min(1.0, 1.0 - loss / 2.2)
            tc.log(loss, accuracy, args.lr)  # one value per column, one row per step

            # One value per dashboard variable, in the order given to
            # init_training_dashboard. A NaN or infinite value is shown as a
            # gap ("NaN") instead of raising, so a diverging run keeps going.
            tc.update_training_dashboard(loss, val_loss, accuracy, args.lr)
            # The hero text is any string you like; set it whenever it changes.
            # This sample uses one step per epoch.
            tc.update_hero_text(
                f"Training at Epoch {epoch}/{args.epochs}, step {epoch}, "
                f"Total Loss at {loss:.4f}"
            )

            metrics.write(
                json.dumps({"step": epoch, "loss": loss, "accuracy": accuracy}) + "\n"
            )
            metrics.flush()

            # Save a checkpoint: raw bytes in any format, whether this is the best
            # so far, plus the step and metrics shown by /checkpoints. Returns the
            # stored path, or None if the settings chose not to store it.
            is_best = loss < best_loss
            best_loss = min(best_loss, loss)
            state = {
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "weights": [random.random() for _ in range(2000)],
            }
            tc.save_checkpoint(
                pickle.dumps(state),
                is_best,
                step=epoch,
                metrics={"loss": loss, "accuracy": accuracy},
            )

            # One line per epoch into several panels at once, so every panel
            # visibly updates while you watch it.
            print(f"processing batches for epoch {epoch}...")  # -> other
            print(f"[log] epoch {epoch}/{args.epochs} finished")  # -> log
            print(  # -> metrics (custom panel)
                f"[metrics] epoch={epoch} loss={loss:.4f} accuracy={accuracy:.4f}"
            )
            print(f"[text_example] heartbeat {epoch}")  # -> text_example
            if args.verbose:
                print(f"  verbose: lr={args.lr} seed={args.seed}")  # -> other
            if args.simulate_errors and epoch % 3 == 0:
                print(  # -> error (red)
                    f"[error] simulated transient failure at epoch {epoch}; retrying"
                )
            time.sleep(args.delay)

            # /pause was called: this epoch's checkpoint is stored, so stop here.
            # Exiting with code 0 makes the controller record the run as paused.
            if pause_requested.is_set() and epoch < args.epochs:
                break

    if pause_requested.is_set() and epoch < args.epochs:
        tc.update_hero_text(f"Paused '{args.name}' after epoch {epoch}, loss {loss:.4f}")
        print(f"[log] run '{args.name}' paused after epoch {epoch}; /resume continues at epoch {epoch + 1}")
        return

    # Final hero text. The dashboard's clock stops by itself shortly after the
    # script exits (it notices the script is gone), no "finished" call needed.
    tc.update_hero_text(f"Finished '{args.name}': {args.epochs} epochs, final loss {loss:.4f}")
    print(f"[log] run '{args.name}' finished")
    tc.test(f"Run '{args.name}' finished; metrics written to {metrics_path.name}")


if __name__ == "__main__":
    main()
