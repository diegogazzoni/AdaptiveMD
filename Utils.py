import matplotlib.pyplot as plt
import numpy as np

def showHistogram(trajectories):
    pass

def showPotential(potential, areaToShow):
    if len(areaToShow) > 2: # >2D potential domain
        print("Impossible to show potential with >2D domain");
        return
    
    fig, ax = plt.subplots()
    fig.suptitle(f'Potential "{potential.name}" surface')
    if len(areaToShow) == 1: # 1D potential
        xtest = np.arange(areaToShow[0][0], areaToShow[0][1], 0.025)
        ax.plot(xtest, potential.surface(xtest))
    else: # 2D potential
        xtest = np.arange(areaToShow[0][0], areaToShow[0][1], 0.025)
        ytest = np.arange(areaToShow[1][0], areaToShow[1][1], 0.025)
        surf = np.zeros_like(xtest.reshape(-1, 1) @ ytest.reshape(1, -1)).T
        for j, x in enumerate(xtest):
            for i, y in enumerate(ytest):
                surf[i, j] = potential.surface([x, y]);
        im = ax.imshow(surf, cmap='viridis', extent=sum(areaToShow, []))
        ax.invert_yaxis()
        ax.vlines(0, areaToShow[1][0], areaToShow[1][1], color='white', linewidth=0.5)
        ax.hlines(0, areaToShow[0][0], areaToShow[0][1], color='white', linewidth=0.5)
        plt.colorbar(im, ax=ax)
