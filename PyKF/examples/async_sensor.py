#!/usr/bin/env python

from PyKF import kf
import numpy as np
import matplotlib.pyplot as plt

# n = number of states
# m = number of observations

def static_eq():
    # Set n x 1 initial state matrix X
    kf.setX([[0],
              [0]])

    # Set n x n P matrix
    kf.setP([[0.1,0],
              [0,0.1]])

    # Set n x n Q matrix
    kf.setQ([[0.0001,0],
              [0,0.0001]])

    # Set m x m R matrix
    kf.setR([[0.04,0,0],
              [0,0.25,0],
              [0,0,0.01]])

    # Set control matrix
    kf.setB([[0]])
    kf.setU([[0]])

def dynamic_eq(t, hy, hy1, hv):
    # Set n x n A matrix
    kf.setA([[1,t],
             [0,1]])

    # Set m x n H matrix
    kf.setH([[hy,0],
             [hy1,0],
             [0,hv]])

kf = kf()
static_eq()

time1 = np.arange(0, 60, 0.1)
time2 = np.arange(0, 60, 0.5)
time = np.unique(np.concatenate((time1,time2),0))
y = np.sin(2*np.pi/60*time2) + np.random.normal(0, 0.2, len(time2))
y1 = np.sin(2*np.pi/60*time2) + np.random.normal(0, 0.5, len(time2))
v = 2*np.pi/60*np.cos(2*np.pi/60*time1) + np.random.normal(0, 0.1, len(time1))
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
time1i = 0
time2i = 0
z_y = 0
z_y1 = 0
z_v = 0
for i in range(len(time)):
    hy = 0
    hy1 = 0
    hv = 0
    t = float(time[i]) - prev
    prev = float(time[i])
    if time1i < len(time1):
        if time1[time1i] == time[i]:
            z_v = v[time1i]
            hv = 1
            time1i += 1
    if time2i < len(time2):
        if time2[time2i] == time[i]:
            z_y = y[time2i]
            z_y1 = y1[time2i]
            hy = 1
            hy1 = 1
            time2i += 1
    z = [[z_y], # Sensor observations m x 1 matrix
         [z_y1],
         [z_v]]
    kf.step(z, dynamic_eq, t, hy, hy1, hv) # KF cycle
    x_filtered = kf.getX() # Get filtered state
    p_filtered = kf.getP() # Get filtered covariance
    g_filtered = kf.getG() # Get Kalman gain
    y_f.append(x_filtered[0,0])
    v_f.append(x_filtered[1,0])
    py.append(p_filtered[0,0])
    pv.append(p_filtered[1,1])
    gy1.append(g_filtered[0,0])
    gy2.append(g_filtered[0,1])
    gv.append(g_filtered[1,2])

yf = np.array(y_f)
plt.plot(time,y_f,'r',label='KF')
plt.plot(time2,y,'g',label='Raw 1')
plt.plot(time2,y1,'b',label='Raw 2')
plt.plot(time,yr,'black',label='Truth')
plt.legend()
plt.title('Displacement')
plt.figure()
plt.plot(time,v_f,'r',label='KF')
plt.plot(time1,v,'g',label='Raw')
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
