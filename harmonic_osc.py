from Verlet import verlet
import time

# Example: harmonic oscillator with acceleration = -k/m * x
k, m = 1.0, 1.0
a_func = lambda x: -(k/m) * x

x0 = 1.0   # initial position
v0 = 0.0   # initial velocity
dt = 0.01
steps = 10000

start = time.time() * 1e3 
xs, vs, ts = verlet(x0, v0, a_func, dt, steps)
stop = time.time()*1e3

print(f'Time elapsed = {(stop - start):.2f} ms')

import matplotlib.pyplot as plt
plt.plot(ts, xs, label="Position")
plt.plot(ts, vs, label="Velocity")
plt.legend()
plt.show()