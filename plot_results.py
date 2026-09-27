"""
Reads log/log.txt (written by train_gpt2.py) and plots train/val loss.
Run this AFTER training finishes:
    python plot_results.py
Saves loss_plot.png in the current directory.
"""

import matplotlib.pyplot as plt

train_steps, train_losses = [], []
val_steps, val_losses = [], []

with open("log/log.txt") as f:
    for line in f:
        parts = line.split()
        if len(parts) != 3:
            continue
        step, tag, loss = int(parts[0]), parts[1], float(parts[2])
        if tag == "train":
            train_steps.append(step)
            train_losses.append(loss)
        elif tag == "val":
            val_steps.append(step)
            val_losses.append(loss)

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, label="train loss", color="tab:blue", alpha=0.6, linewidth=1)
plt.plot(val_steps, val_losses, label="val loss", color="tab:orange", marker="o", linewidth=2)
plt.xlabel("steps")
plt.ylabel("loss")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=150)
print("Saved loss_plot.png")