import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import animation
import pdb
from math import atan2
from tqdm import tqdm

if __name__ == "__main__":
    p_occ = .7
    p_free = 1 - p_occ

    alpha = 1.0 # meters
    beta = np.deg2rad(5)
    z_max = 150 # meters

    # initially, we know nothing about each cell, so p(m_i) = .5, which means
    # that p(m_i) / 1 - p(m_i) = .5 / .5 = 1, so log(1) = 0 ... this makes sense
    # since we have no prior information about each cell in the map
    l_0 = 0

    l_occ = np.log(p_occ / (1 - p_occ))
    l_free = np.log(p_free / (1 - p_free))

    world = np.zeros((100,100))
    world[:,:] = l_0

    # get poses and measurements
    mat_file = loadmat('state_meas_data.mat')
    th_k = mat_file['thk']
    states = mat_file['X']
    z = mat_file['z']

    times = [i for i in range(states.shape[1])]

    # subtract 1 from pose loaction coords b/c python is 0-based indexing and matlab is 1-based
    states[0,:] -= 1
    states[1,:] -= 1
    # make the origin of the world be in the bottom left corner
    # states[1,:] = (world.shape[0] - 1) - states[1,:]

    '''
    # TODO see if this is necessary ... maybe just make the range 500 (something large)
    # make NaN ranges infinity
    z[0,:,:] = np.where(np.isnan(z[0,:,:]), np.inf, z[0,:,:])
    # make NaN angles the regular angle
    for t in times:
        z[1,:,t] = th_k[0,:]
    '''

    '''
    for t in times:
        t=61
        x = states[0,t]
        y = states[1,t]
        orientation = states[2,t]
        print("x:", x)
        print("y:", y)
        print("th:", orientation)

        r = z[0,0,t]
        th = orientation + z[1,0,t]
        print("\nmeasurement at", th, "radians")
        print("r:", r)
        print("new x:", x + ( np.cos(th) * r) )
        print("new y:", y - ( np.sin(th) * r) )

        r = z[0,5,t]
        th = orientation + z[1,5,t]
        print("\nmeasurement at", th, "radians")
        print("r:", r)
        print("new x:", x + ( np.cos(th) * r) )
        print("new y:", y - ( np.sin(th) * r) )

        r = z[0,-1,t]
        th = orientation + z[1,-1,t]
        print("\nmeasurement at", th, "radians")
        print("r:", r)
        print("new x:", x + ( np.cos(th) * r) )
        print("new y:", y - ( np.sin(th) * r) )

        unit_vec = ( x + np.cos(orientation) , y - np.sin(orientation) )
        print("\nunit vector in the direction of orientation:")
        print(unit_vec)

        break
    '''

    '''
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    ims = []
    for t in times:
        if t > 5:
            break
        x = int(states[0,t])
        y = int(states[1,t])
        world[y,x] = 1
        im = plt.imshow(world, cmap='gray')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat=False) 
    plt.pause(.1)
    input("<Hit enter to close>")
    '''

    min_perception_bound = np.amin(th_k)
    max_perception_bound = np.amax(th_k)

    loop = tqdm(total=len(times), position=0)
    
    fig = plt.figure()
    ax = plt.gca()
    ax.set_yticks([])
    ax.set_xticks([])
    ims = []
    im = plt.imshow(world, cmap='gray')
    ims.append([im])
    for t in times:
        # for all cells in the world
        for y in range(world.shape[0]):
            for x in range(world.shape[1]):
                if (x == states[0,t]) and (y == states[1,t]):
                    continue # don't update the cell that the robot is in

                x_diff = x - states[0,t]
                y_diff = y - states[1,t]
                bearing = np.arctan2(y_diff, x_diff) - states[2,t]

                # make sure cell is in perceptual field of view
                if (bearing < min_perception_bound) or (bearing > max_perception_bound):
                    continue

                # inverse range sensor model
                r = np.sqrt( (x_diff * x_diff) + (y_diff * y_diff) )
                k = np.argmin( np.abs(bearing - th_k) )
                z_k_t = z[0,k,t]
                th_k_sens = th_k[0,k]
                update_amt = l_0
                if ( r > min(z_max, z_k_t + (alpha/2)) ) or ( np.abs(bearing - th_k_sens) > (beta/2) ):
                    update_amt = l_0
                elif (z_k_t < z_max) and ( np.abs(bearing - th_k_sens) > (beta/2) ):
                    update_amt = l_occ
                elif r <= z_k_t:
                    update_amt = l_free

                world[y,x] += update_amt - l_0

        # show where the robot is
        x_idx, y_idx = int(states[0,t]), int(states[1,t])
        temp = world[y_idx, x_idx]
        world[y_idx, x_idx] = np.amin(world)

        # animate
        # (flipud makes the world origin start from bottom left instead of top left)
        im = plt.imshow(np.flipud(world * -1), cmap='gray')
        ims.append([im])

        # reset robot cell to original
        world[y_idx, x_idx] = temp

        loop.update(1)

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat=False) 
    plt.pause(.1)
    input("<Hit enter to close>")