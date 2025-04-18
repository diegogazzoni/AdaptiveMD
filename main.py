import numpy as np
import matplotlib.pyplot as plt

from BrownianEngine import *
from Potential import *
import Utils as ut

if __name__ == '__main__':
    sides = [[-1.25, 0.75], [-0.15, 2.0]]
    xbins = np.arange(sides[0][0], sides[0][1], 0.05)
    ybins = np.arange(sides[1][0], sides[1][1], 0.05)

    mullerBrown = MullerBrown()
    ut.showPotential(potential=mullerBrown, areaToShow=sides)

    trajectories = []
    timevecs = []
    for i in range(1):
        brownian = BrownianParticleSimulator(mullerBrown, sides, kT=0.05, pbc=True)
        startPos = np.array((np.random.uniform(sides[0][0], sides[0][1]), np.random.uniform(sides[1][0], sides[1][1])))
        brownian.run(startPos, 10000)
        trajectories.append( np.vstack(brownian.P) )
        timevecs.append( brownian.timestep )

    fig, ax = plt.subplots(2)
    fig.suptitle('Trajectories and histogram')
    for time, traj in zip(timevecs, trajectories):
        if traj.shape[0] > 1:
            #ax[0].plot(traj[1, 0], traj[1, 1], '.', linewidth=0.5 )
            ax[0].plot(traj[:, 0], traj[:, 1], linewidth=0.5)
            ax[0].set_xlim(sides[0][0], sides[0][1])
            ax[0].set_ylim(sides[1][0], sides[1][1])
            ax[1].hist2d(np.vstack(trajectories)[:, 0], np.vstack(trajectories)[:, 1], bins=(xbins, ybins))
        else:
            ax[0].plot(time, traj[:, 0], traj[:, 1])
    
    plt.show()