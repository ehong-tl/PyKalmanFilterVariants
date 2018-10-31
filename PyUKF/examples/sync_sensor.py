#!/usr/bin/env python

from PyUKF import ukf
import numpy as np
import matplotlib.pyplot as plt

# n = number of states
# m = number of observations

def static_eq():
    # Set n x 1 initial state matrix X
    ukf.setX([[0],
              [0]])

    # Set n x n P matrix
    ukf.setP([[0.1,0],
              [0,0.1]])

    # Set n x n Q matrix
    ukf.setQ([[0.0001,0],
              [0,0.0001]])

    # Set m x m R matrix
    ukf.setR([[0.04,0,0],
              [0,0.25,0],
              [0,0,0.01]])

    ukf.setAlpha(0.0001)
    ukf.setBeta(2)
    ukf.setKappa(0)

def dynamic_eq(t):
    # Set n x 1 f(x) matrix and m x 1 h(x) matrix
    fx = [[ukf.xs[ukf.i][0,0] + t * ukf.xs[ukf.i][1,0]],
          [ukf.xs[ukf.i][1,0]]]
    hx = [[ukf.xs[ukf.i][0,0]],
          [ukf.xs[ukf.i][0,0]],
          [ukf.xs[ukf.i][1,0]]]
    return fx, hx

ukf = ukf()
static_eq()

time = np.arange(0, 60, 0.1)
y = np.sin(2*np.pi/60*time) + np.random.normal(0, 0.2, len(time))
y1 = np.sin(2*np.pi/60*time) + np.random.normal(0, 0.5, len(time))
v = 2*np.pi/60*np.cos(2*np.pi/60*time) + np.random.normal(0, 0.1, len(time))
yr = np.sin(2*np.pi/60*time)
vr = 2*np.pi/60*np.cos(2*np.pi/60*time)

prev = 0
y_f = []
v_f = []
py = []
pv = []
gy1 = []
gy2 = []
gv = []
for i in range(len(time)):
    t = float(time[i]) - prev
    prev = float(time[i])
    z = [[y[i]], # Sensor observations m x 1 matrix
         [y1[i]],
         [v[i]]]
    ukf.step(z, dynamic_eq, t) # UKF cycle
    x_filtered = ukf.getX() # Get filtered state
    p_filtered = ukf.getP() # Get filtered covariance
    g_filtered = ukf.getG() # Get Kalman gain
    y_f.append(x_filtered[0,0])
    v_f.append(x_filtered[1,0])
    py.append(p_filtered[0,0])
    pv.append(p_filtered[1,1])
    gy1.append(g_filtered[0,0])
    gy2.append(g_filtered[0,1])
    gv.append(g_filtered[1,2])

yf = np.array(y_f)
plt.plot(time,y_f,'r',label='UKF')
plt.plot(time,y,'g',label='Raw 1')
plt.plot(time,y1,'b',label='Raw 2')
plt.plot(time,yr,'black',label='Truth')
plt.legend()
plt.title('Displacement')
plt.figure()
plt.plot(time,v_f,'r',label='UKF')
plt.plot(time,v,'g',label='Raw')
plt.plot(time,vr,'black',label='Truth')
plt.legend()
plt.title('Velocity')
plt.figure()
py = np.array(py)
pv = np.array(pv)
plt.plot(time, py, 'r',label='Displacement')
plt.plot(time, pv, 'g',label='Velocity')
plt.legend()
plt.title('Covariance')
plt.figure()
gy1 = np.array(gy1)
gy2 = np.array(gy2)
gv = np.array(gv)
plt.plot(time, gy1, 'r',label='Displacement 1')
plt.plot(time, gy2, 'g',label='Displacement 2')
plt.plot(time, gv, 'b',label='Velocity')
plt.legend()
plt.title('Kalman Gain')
plt.show()
