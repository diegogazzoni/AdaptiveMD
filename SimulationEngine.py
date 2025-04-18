class SimulationEngine(object):
    def __init__(self):
        self.steps = 0
        self.name = 'Engine'

    def run(self, X, numPoints):
        raise NotImplementedError()