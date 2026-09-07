"""

Generates the "ground truth" 1D trajectory that the Kalman filter will
later try to estimate from noisy sensor data. No noise is added here --
this is the reference we compare filter output against.

Motion profile: piecewise-constant acceleration, chosen to give the
trajectory some real dynamics (not just constant velocity), so the
filter has something meaningful to track.
"""

import numpy as np


def generate_ground_truth(dt=0.01, total_time=20.0):
    """
    Generate ground truth position, velocity, and acceleration over time.

    Motion profile (roughly):
      0s  -  4s : accelerate from rest at 1.0 m/s^2
      4s  - 10s : cruise at constant velocity (accel = 0)
      10s - 13s : decelerate at -1.5 m/s^2
      13s - 16s : cruise at the new (lower) constant velocity
      16s - 20s : accelerate again at 0.8 m/s^2

    Parameters
    ----------
    dt : float
        Timestep in seconds.
    total_time : float
        Total duration of the simulation in seconds.

    Returns
    -------
    t : np.ndarray, shape (N,)
    pos : np.ndarray, shape (N,)   -- true position (m)
    vel : np.ndarray, shape (N,)   -- true velocity (m/s)
    acc : np.ndarray, shape (N,)   -- true acceleration (m/s^2), for reference
    """
    t = np.arange(0, total_time, dt)
    n = len(t)

    acc = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < 4.0:
            acc[i] = 1.0
        elif ti < 10.0:
            acc[i] = 0.0
        elif ti < 13.0:
            acc[i] = -1.5
        elif ti < 16.0:
            acc[i] = 0.0
        else:
            acc[i] = 0.8

    pos = np.zeros(n)
    vel = np.zeros(n)

    # Integrate acceleration -> velocity -> position using simple Euler
    # integration. dt is small enough (0.01s) that this is accurate
    # enough for a ground truth reference.
    for i in range(1, n):
        vel[i] = vel[i - 1] + acc[i - 1] * dt
        pos[i] = pos[i - 1] + vel[i - 1] * dt + 0.5 * acc[i - 1] * dt**2

    return t, pos, vel, acc


if __name__ == "__main__":
    t, pos, vel, acc = generate_ground_truth()

    print(f"Simulated {len(t)} timesteps over {t[-1]:.2f} seconds")
    print(f"Final position: {pos[-1]:.2f} m")
    print(f"Final velocity: {vel[-1]:.2f} m/s")
    print(f"Max velocity:   {vel.max():.2f} m/s")