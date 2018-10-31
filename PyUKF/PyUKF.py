import numpy as np
from scipy import linalg as scipy_alg

# n = number of states
# m = number of observations

class model:

    def __init__(self):
        self.Xs = None      # 2L+1 sigma points for fx
        self.Zs = None      # 2L+1 sigma points for hx
    
    def setX(self, mat):    # n x 1
        self.x = np.matrix(mat)

    def setfx(self, mat):   # n x 1
        self.fx = np.matrix(mat)

    def sethx(self, mat):   # m x 1
        self.hx = np.matrix(mat)

    def setP(self, mat):    # n x n
        self.P = np.matrix(mat)

    def setQ(self, mat):    # n x n
        self.Q = np.matrix(mat)

    def setR(self, mat):    # m x m
        self.R = np.matrix(mat)

    def setAlpha(self, val):
        self.alpha = val

    def setBeta(self, val):
        self.beta = val

    def setKappa(self, val):
        self.kappa = val    

class ukf(model):

    def __init__(self):
        model.__init__(self)

    def sigma(self):
        points = []
        self.weights_x = []
        self.weights_P = []
        L = self.x.shape[0]
        lam = (self.alpha**2)*(L+self.kappa) - L
        points.append(self.x)
        add_val = scipy_alg.sqrtm((L + lam) * self.P)
        for i in range(add_val.shape[1]):
            add_val_mat = np.zeros(self.P.shape[0])
            for j in range(add_val.shape[0]):
                add_val_mat[j] = add_val[j][i]
            add_val_mat = np.matrix(add_val_mat).T
            points.append(self.x + add_val_mat)
        for i in range(add_val.shape[1]):
            add_val_mat = np.zeros(self.P.shape[0])
            for j in range(add_val.shape[0]):
                add_val_mat[j] = add_val[j][i]
            add_val_mat = np.matrix(add_val_mat).T
            points.append(self.x - add_val_mat)
        self.weights_x.append(lam/(L+lam))
        self.weights_P.append((lam/(L+lam)) + (1-self.alpha**2+self.beta))
        for i in range(2*L):
            self.weights_x.append(1/(2*(L+lam)))
            self.weights_P.append(1/(2*(L+lam)))
        self.xs = points

    def update_sigma(self, func, *args):
        self.sigma()
        self.Xs = []
        self.Zs = []
        for self.i in range(len(self.xs)):
            Xs_tem, Zs_tem = func(*args)
            self.Xs.append(np.matrix(Xs_tem))
            self.Zs.append(np.matrix(Zs_tem))

    def predict(self):
        self.xp = 0
        self.zp = 0
        for i in range(len(self.weights_x)):
            self.xp += self.weights_x[i] * self.Xs[i]
            self.zp += self.weights_x[i] * self.Zs[i]
        self.Pp = 0
        for i in range(len(self.weights_P)):
            self.Pp += self.weights_P[i] * (self.Xs[i]-self.xp)*(self.Xs[i]-self.xp).T
        self.Pp += self.Q
        
    def correct(self):
        self.Pyy = 0
        self.Pxy = 0
        for i in range(len(self.weights_P)):
            self.Pyy += self.weights_P[i] * (self.Zs[i]-self.zp)*(self.Zs[i]-self.zp).T
            self.Pxy += self.weights_P[i] * (self.Xs[i]-self.xp)*(self.Zs[i]-self.zp).T
        self.Pyy += self.R
        self.G = self.Pxy*np.linalg.pinv(self.Pyy)
        self.x = self.xp + self.G*(self.z - self.zp)
        self.P = self.Pp - self.G*self.Pyy*self.G.T

    def step(self, z, update_func, *args):
        self.update_sigma(update_func, *args)
        self.z = np.matrix(z)
        self.predict()
        self.correct()
        return 0

    def getX(self):
        return self.x

    def getP(self):
        return self.P

    def getG(self):
        return self.G
