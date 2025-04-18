import numpy as np
from SimulationEngine import *

class BrownianParticleSimulator(SimulationEngine):
    def __init__(self, potential, domainSides, mass=1.0, damping=1.0, kT=1.0, dt=0.01, pbc=False):
        super().__init__();
        self.potential = potential
        self.mass = mass
        self.damping = damping
        self.kT = kT
        self.dt = dt
        self.domainSides = domainSides # couples of ((inf, sup), (inf, sup))
        self.dimensions = len( domainSides )
        self.P = []
        self.timestep = []
        self.applyPbc = pbc
        self.name = 'Brownian particle simulator'

        self.coeffExt = self.dt / (self.mass * self.damping)
        self.coeffStoc = np.sqrt(2.0 * self.dt * self.kT / (self.mass * self.damping))
    
    def run(self, X0, numPoints):
        if not isinstance(X0, np.ndarray):
            raise ValueError("X must be a numpy array.");

        if X0.shape[0] != self.dimensions:
            raise ValueError("Incorrect dimensions of X. They must be consistent with potential domain!");
    
        self.P.append( X0 )
        self.timestep.append( 0 )
        for it in range(1, numPoints+1):
            randomVector = np.random.normal( size=X0.shape[0] )
            newPosition = self.P[-1] - self.coeffExt*self.potential.gradient(self.P[-1]) + self.coeffStoc*randomVector
            if self.applyPbc:
                for d, box in enumerate(self.domainSides):
                    dim = (box[1] - box[0])
                    newPosition[d] = (newPosition[d] - box[0]) % dim + box[0]

            self.P.append( newPosition )
            self.timestep.append( it*self.dt )
            self.steps += 1
            
