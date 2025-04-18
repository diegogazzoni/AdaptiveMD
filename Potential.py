import numpy as np

class Potential(object):
    def surface(self, X):
        self.name = 'Potential'
        raise NotImplementedError()

    def gradient(self, X):
        raise NotImplementedError()

class DoubleWell1D(Potential):
    def __init__(self, c=1.0, X0=0):
        self.c = c;
        self.X0 = X0
        self.name = 'Double Well 1D'

    def surface(self, X):
        return 0.25*(X-self.X0)**4 - (X-self.X0)**2 - self.c*(X-self.X0)
    
    def gradient(self, X):
        return 0.75*(X-self.X0)**3 - 2*(X-self.X0) - self.c

class DoubleWell2D(Potential):
    def __init__(self, a_x=1.0, a_y=0.5, b_x=1.0, b_y=1.5, center=(0.0, 0.0)):
        self.a_x = a_x
        self.a_y = a_y
        self.b_x = b_x
        self.b_y = b_y
        self.x0, self.y0 = center
        self.name = 'Double Well 2D'

    def surface(self, X):
        x, y = X
        x_shifted = x - self.x0
        y_shifted = y - self.y0
        Vx = self.a_x * (x_shifted**2 - self.b_x**2)**2
        Vy = self.a_y * (y_shifted**2 - self.b_y**2)**2
        return Vx + Vy

    def gradient(self, X):
        x, y = X
        x_shifted = x - self.x0
        y_shifted = y - self.y0
        dVx = 4 * self.a_x * x_shifted * (x_shifted**2 - self.b_x**2)
        dVy = 4 * self.a_y * y_shifted * (y_shifted**2 - self.b_y**2)
        return np.array([dVx, dVy])

class MullerBrown(Potential):
    def __init__(self, A=(-2,-1,-1.7,0.15), a=(-1,-1,-6.5,0.7), b=(0,0,11,0.6), c=(-10,-10,-6.5,0.7), x0=(1,0,-0.5,-1), y0=(0, 0.5, 1.5, 1)):
        self.A = A
        self.a = a
        self.b = b
        self.c = c
        self.x0 = x0
        self.y0 = y0
        self.name = 'Muller Brown'

    def surface(self, X):
        x, y = X
        return np.sum( [self.A[i]*np.exp( self.a[i]*(x-self.x0[i])**2 + self.b[i]*(x-self.x0[i])*(y-self.y0[i]) + self.c[i]*(y-self.y0[i])**2) for i in range(4)] )

    def gradient(self, X):
        x, y = X
        dx = 0.0
        dy = 0.0
        for i in range(4):
            dx_i = 2 * self.a[i] * (x - self.x0[i]) + self.b[i] * (y - self.y0[i])
            dy_i = self.b[i] * (x - self.x0[i]) + 2 * self.c[i] * (y - self.y0[i])
            exp_term = np.exp(
                self.a[i] * (x - self.x0[i])**2 +
                self.b[i] * (x - self.x0[i]) * (y - self.y0[i]) +
                self.c[i] * (y - self.y0[i])**2
            )
            dx += self.A[i] * exp_term * dx_i
            dy += self.A[i] * exp_term * dy_i
        return np.array([dx, dy]) # removed -