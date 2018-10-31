import numpy as np

# n = number of states
# m = number of observations

class model:

    def __init__(self):
        pass
    
    def setX(self, mat):    # n x 1
        self.x = np.matrix(mat)

    def setA(self, mat):    # n x n
        self.A = np.matrix(mat)

    def setB(self, mat):
        self.B = np.matrix(mat)

    def setU(self, mat):
        self.u = np.matrix(mat)

    def setH(self, mat):    # m x n
        self.H = np.matrix(mat)

    def setP(self, mat):    # n x n
        self.P = np.matrix(mat)

    def setQ(self, mat):    # n x n
        self.Q = np.matrix(mat)

    def setR(self, mat):    # m x m
        self.R = np.matrix(mat)

class kf(model):

    def __init__(self):
        model.__init__(self)

    def predict(self):
        self.xp = self.A*self.x + self.B*self.u
        self.Pp = self.A*self.P*self.A.T + self.Q
        
    def correct(self):
        self.G = self.Pp*self.H.T*np.linalg.pinv(self.H*self.Pp*self.H.T + self.R)
        self.x = self.xp + self.G*(self.z - self.H*self.xp)
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
