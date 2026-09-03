"""
plot_truth.py

Quick visual sanity check of the ground truth trajectory: plots
position, velocity, and acceleration over time so we can confirm
the motion profile looks right before building anything on top of it.
"""

import matplotlib.pyplot as plt
from simulate_truth import generate_ground_truth


def plot_ground_truth():
    t, pos, vel, acc = generate_ground_truth()

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t, pos, color="tab:blue")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Ground Truth Trajectory")
    axes[0].grid(True)

    axes[1].plot(t, vel, color="tab:orange")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True)

    axes[2].plot(t, acc, color="tab:green")
    axes[2].set_ylabel("Acceleration (m/s^2)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("ground_truth.png", dpi=150)
    print("Saved plot to ground_truth.png")
    plt.show()


if __name__ == "__main__":
    plot_ground_truth()