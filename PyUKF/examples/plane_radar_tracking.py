#!/usr/bin/env python

# http://www.goddardconsulting.ca/simulink-extended-kalman-filter-tracking.html

from PyUKF import ukf
import numpy as np
import matplotlib.pyplot as plt
import math

# n = number of states
# m = number of observations

def static_eq():
    # Set n x 1 initial state matrix X
    ukf.setX([[-2900],
              [80],
              [950],
              [20]])

    # Set n x n P matrix
    ukf.setP([[100,0,0,0],
              [0,100,0,0],
              [0,0,100,0],
              [0,0,0,100]])

     # Set n x n Q matrix
    ukf.setQ([[0,0,0,0],
              [0,0.1,0,0],
              [0,0,0,0],
              [0,0,0,0.1]])

    # Set m x m R matrix
    ukf.setR([[50**2,0],
              [0,0.005**2]])

    ukf.setAlpha(0.001)
    ukf.setBeta(2)
    ukf.setKappa(0)

def dynamic_eq(t):
    # Set n x 1 f(x) matrix and m x 1 h(x) matrix
    fx = [[ukf.xs[ukf.i][0,0] + t * ukf.xs[ukf.i][1,0]],
          [ukf.xs[ukf.i][1,0]],
          [ukf.xs[ukf.i][2,0] + t * ukf.xs[ukf.i][3,0]],
          [ukf.xs[ukf.i][3,0]]]
    hx = [[math.sqrt(ukf.xs[ukf.i][0,0]**2 + ukf.xs[ukf.i][2,0]**2)],
          [math.atan2(ukf.xs[ukf.i][2,0],ukf.xs[ukf.i][0,0])]]
    return fx, hx

ukf = ukf()
static_eq()

time = np.arange(0, 60, 0.1)
x = np.linspace(-3000,3000,len(time))
y = np.linspace(1000,800,len(time))
xv = np.gradient(x,time)
yv = np.gradient(y,time)
r_raw = np.sqrt(x**2+y**2)
theta_raw = np.arctan2(y,x)
r_noise = np.random.normal(0,50,len(r_raw))
theta_noise = np.random.normal(0,0.005,len(theta_raw))
r = r_raw + r_noise
theta = theta_raw + theta_noise

prev = 0
x_f = []
y_f = []
xv_f = []
yv_f = []
for i in range(len(time)):
    t = float(time[i]) - prev
    prev = float(time[i])
    z = [[r[i]], # Sensor observations m x 1 matrix
         [theta[i]]]
    ukf.step(z, dynamic_eq, t) # UKF cycle
    x_filtered = ukf.getX() # Get filtered state
    p_filtered = ukf.getP() # Get filtered covariance
    g_filtered = ukf.getG() # Get Kalman gain
    x_f.append(x_filtered[0,0])
    xv_f.append(x_filtered[1,0])
    y_f.append(x_filtered[2,0])
    yv_f.append(x_filtered[3,0])

x_f = np.array(x_f)
xv_f = np.array(xv_f)
y_f = np.array(y_f)
yv_f = np.array(yv_f)
plt.subplot(411)
plt.plot(time,x,'b',label='Truth')
plt.plot(time,x_f,'r',label='UKF')
plt.title('X position')
plt.xticks([])
plt.subplot(412)
plt.plot(time,xv,'b',label='Truth')
plt.plot(time,xv_f,'r',label='UKF')
plt.title('X velocity')
plt.xticks([])
plt.subplot(413)
plt.plot(time,y,'b',label='Truth')
plt.plot(time,y_f,'r',label='UKF')
plt.title('Y position')
plt.xticks([])
plt.subplot(414)
plt.plot(time,yv,'b',label='Truth')
plt.plot(time,yv_f,'r',label='UKF')
plt.title('Y velocity')
plt.legend()
plt.figure()
plt.plot(x,y,'b',label='Truth')
plt.plot(x_f,y_f,'r',label='UKF')
plt.title('XY Position')
plt.legend()
plt.show()
