"""
Runs the Kalman filter over the simulated sensor data and compares
the filter's estimate against ground truth, GPS-only, and IMU-only
approaches.
"""

import numpy as np
import matplotlib.pyplot as plt

from simulate_truth import generate_ground_truth
from simulate_sensors import generate_sensor_data
from kalman_filter import KalmanFilter1D


def main():
    # Generate ground truth and noisy sensor data
    t, pos, vel, acc = generate_ground_truth()
    t, gps_pos, imu_acc, bias = generate_sensor_data(t, pos, acc, seed=42)

    dt = t[1] - t[0]
    n = len(t)

    # Initialize the filter
    kf = KalmanFilter1D(
        dt=dt,
        x0=[0, 0],
        P0=np.eye(2) * 1.0,
        sigma_a=0.05,
        sigma_gps=0.5
    )

    # Run predict/update every timestep (GPS assumed available every step for now)
    est_pos = np.zeros(n)
    est_vel = np.zeros(n)

    for i in range(n):
        kf.predict(imu_acc[i])
        kf.update(gps_pos[i])
        est_pos[i] = kf.x[0, 0]
        est_vel[i] = kf.x[1, 0]

    # Plot: true position vs GPS-only vs filter estimate
    plt.figure(figsize=(10, 6))
    plt.plot(t, pos, label="True position", color="black", linewidth=1.5)
    plt.scatter(t, gps_pos, label="GPS (raw)", color="tab:blue", s=2, alpha=0.3)
    plt.plot(t, est_pos, label="Kalman filter estimate", color="tab:red", linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Kalman Filter Position Estimate vs Ground Truth and Raw GPS")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../results/kalman_filter_result.png", dpi=150)
    plt.show()

    # Print error metrics
    gps_error = np.abs(gps_pos - pos)
    filter_error = np.abs(est_pos - pos)

    print(f"GPS-only mean abs error:    {gps_error.mean():.4f} m")
    print(f"Filter mean abs error:      {filter_error.mean():.4f} m")
    print(f"GPS-only max abs error:     {gps_error.max():.4f} m")
    print(f"Filter max abs error:       {filter_error.max():.4f} m")


if __name__ == "__main__":
    main()