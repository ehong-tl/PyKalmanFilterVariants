#!/usr/bin/env python

# http://www.goddardconsulting.ca/simulink-extended-kalman-filter-tracking.html

from PyEKF import ekf
import numpy as np
import matplotlib.pyplot as plt
import math

# n = number of states
# m = number of observations

def static_eq():
    # Set n x 1 initial state matrix X
    ekf.setX([[-2900],
              [80],
              [950],
              [20]])

    # Set n x n P matrix
    ekf.setP([[100,0,0,0],
              [0,100,0,0],
              [0,0,100,0],
              [0,0,0,100]])

    # Set n x n Q matrix
    ekf.setQ([[0,0,0,0],
              [0,0.1,0,0],
              [0,0,0,0],
              [0,0,0,0.1]])

    # Set m x m R matrix
    ekf.setR([[50**2,0],
              [0,0.005**2]])

def dynamic_eq(t, z):
    # Set n x 1 f(x) matrix
    ekf.setfx([[ekf.x[0,0] + t * ekf.x[1,0]],
               [ekf.x[1,0]],
               [ekf.x[2,0] + t * ekf.x[3,0]],
               [ekf.x[3,0]]])
    
    # Set n x n Jacobian matrix F
    ekf.setF([[1,t,0,0],
              [0,1,0,0],
              [0,0,1,t],
              [0,0,0,1]])

    # Set m x n Jacobian matrix H
    ekf.setH([[math.cos(z[1,0]),0,math.sin(z[1,0]),0],
              [-math.sin(z[1,0])/z[0,0],0,math.cos(z[1,0])/z[0,0],0]])

    # Set m x 1 h(x) matrix
    ekf.sethx([[math.sqrt(ekf.x[0,0]**2 + ekf.x[2,0]**2)],
               [math.atan2(ekf.x[2,0],ekf.x[0,0])]])

ekf = ekf()
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
    ekf.step(z, dynamic_eq, t, np.matrix(z)) # EKF cycle
    x_filtered = ekf.getX() # Get filtered state
    p_filtered = ekf.getP() # Get filtered covariance
    g_filtered = ekf.getG() # Get Kalman gain
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
plt.plot(time,x_f,'r',label='EKF')
plt.title('X position')
plt.xticks([])
plt.subplot(412)
plt.plot(time,xv,'b',label='Truth')
plt.plot(time,xv_f,'r',label='EKF')
plt.title('X velocity')
plt.xticks([])
plt.subplot(413)
plt.plot(time,y,'b',label='Truth')
plt.plot(time,y_f,'r',label='EKF')
plt.title('Y position')
plt.xticks([])
plt.subplot(414)
plt.plot(time,yv,'b',label='Truth')
plt.plot(time,yv_f,'r',label='EKF')
plt.title('Y velocity')
plt.legend()
plt.figure()
plt.plot(x,y,'b',label='Truth')
plt.plot(x_f,y_f,'r',label='EKF')
plt.title('XY Position')
plt.legend()
plt.show()
