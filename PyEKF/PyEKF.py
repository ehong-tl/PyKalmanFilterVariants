import numpy as np

# n = number of states
# m = number of observations

class model:

    def __init__(self):
        pass
    
    def setX(self, mat):    # n x 1
        self.x = np.matrix(mat)

    def setfx(self, mat):   # n x 1
        self.fx = np.matrix(mat)

    def setF(self, mat):    # n x n
        self.F = np.matrix(mat)

    def sethx(self, mat):   # m x 1
        self.hx = np.matrix(mat)

    def setH(self, mat):    # m x n
        self.H = np.matrix(mat)

    def setP(self, mat):    # n x n
        self.P = np.matrix(mat)

    def setQ(self, mat):    # n x n
        self.Q = np.matrix(mat)

    def setR(self, mat):    # m x m
        self.R = np.matrix(mat)

class ekf(model):

    def __init__(self):
        model.__init__(self)

    def predict(self):
        self.Pp = self.F*self.P*self.F.T + self.Q
        
    def correct(self):
        self.G = self.Pp*self.H.T*np.linalg.pinv(self.H*self.Pp*self.H.T + self.R)
        self.x = self.fx + self.G*(self.z - self.hx)
        self.P = (np.identity(self.P.shape[0]) - self.G*self.H)*self.Pp

    def step(self, z, update_func, *args):
        update_func(*args)
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
