import numpy as np

from simulate_truth import generate_ground_truth

def generate_sensor_data(t , pos , acc , gps_noise_std=0.5 , imu_noise_std =0.5 ,imu_bias_walk_std = 0.01, seed=None ):
    if seed is not None:
        np.random.seed(seed)
    n = len(t)
    gps_pos = pos + np.random.normal(0,gps_noise_std,size=n)

    bias = np.zeros(n)
    for i in range(1,n):
        bias[i]=bias[i-1]+np.random.normal(0,imu_bias_walk_std)
    imu_acc = acc + bias + np.random.normal(0,imu_bias_walk_std, size=n)
    return t,gps_pos,imu_acc,bias


if __name__ == "__main__":
    t, pos, vel, acc = generate_ground_truth()
    t, gps_pos, imu_acc, bias = generate_sensor_data(t, pos, acc, seed=42)

    print(f"Simulated {len(t)} timesteps over {t[-1]:.2f} seconds")
    print(f"Max GPS position error:   {np.max(np.abs(gps_pos - pos)):.3f} m")
    print(f"Max IMU accel error:      {np.max(np.abs(imu_acc - acc)):.3f} m/s^2")
    print(f"Final IMU bias:           {bias[-1]:.4f} m/s^2")
    print(f"Max IMU bias magnitude:   {np.max(np.abs(bias)):.4f} m/s^2")