"""Demo of the training dashboard API.

In the REPL:
    /add_scripts demo D:\\training_controller\\training_sample.py
    /open_script demo
    /run_script
    /dashboard training
"""
import math
import random
import time

import training_controller as tc

# (name, type) pairs. Types: "curve" (line plot), "graph" (bars), "box" (latest value).
tc.init_training_dashboard(
    "loss", "curve",
    "val_loss", "curve",
    "accuracy", "graph",
    "lr", "box",
)

step = 0
for epoch in range(1, 51):
    for _ in range(20):
        step += 1
        loss = 2 * math.exp(-step / 200) + 0.002 + random.random() * 0.01
        acc = 1 - 0.9 * math.exp(-step / 250)
        lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * min(step / 1000, 1)))
        # One value per variable, in the order given to init_training_dashboard.
        tc.update_training_dashboard(loss, loss * 1.1, acc, lr)
        time.sleep(0.05)
    tc.update_hero_text(f"Training at Epoch {epoch}, step {step}, Total Loss at {loss:.4f}")
