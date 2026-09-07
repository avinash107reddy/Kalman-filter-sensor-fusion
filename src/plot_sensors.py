"""
Plots the noisy GPS-like position and IMU-like acceleration sensor
readings against the ground truth, so we can visually verify the
sensor simulation looks correct before building the Kalman filter.
"""

import matplotlib.pyplot as plt

from simulate_truth import generate_ground_truth
from simulate_sensors import generate_sensor_data


def main():
    t, pos, vel, acc = generate_ground_truth()
    t, gps_pos, imu_acc, bias = generate_sensor_data(t, pos, acc, seed=42)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Position: true vs GPS
    axes[0].plot(t, pos, label="True position", color="black", linewidth=1.5)
    axes[0].scatter(t, gps_pos, label="GPS reading", color="tab:blue", s=2, alpha=0.4)
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("GPS-like Position Sensor vs Ground Truth")
    axes[0].legend()

    # Acceleration: true vs IMU
    axes[1].plot(t, acc, label="True acceleration", color="black", linewidth=1.5)
    axes[1].plot(t, imu_acc, label="IMU reading", color="tab:red", linewidth=0.8, alpha=0.7)
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].set_title("IMU-like Acceleration Sensor vs Ground Truth")
    axes[1].legend()

    # Bias random walk (ground-truth-only, for debugging/intuition)
    axes[2].plot(t, bias, label="IMU bias (random walk)", color="tab:orange")
    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Bias (m/s^2)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Underlying IMU Bias Drift (not visible to the filter)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()