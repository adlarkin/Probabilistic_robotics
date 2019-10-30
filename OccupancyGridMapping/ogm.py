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
    beta = np.deg2rad(2)
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
    z = np.nan_to_num(z)    # make NaN 0

    times = [i for i in range(states.shape[1])]

    # subtract 1 from pose loaction coords b/c python is 0-based indexing and matlab is 1-based
    states[0,:] -= 1
    states[1,:] -= 1

    min_perception_bound = np.amin(th_k)
    max_perception_bound = np.amax(th_k)

    mapped_world = None

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
                k = np.argmin( np.abs(bearing - z[1,:,t]) )
                z_k_t = z[0,k,t]
                th_k_sens = z[1,k,t]
                update_amt = l_0
                if ( r > min(z_max, z_k_t + (alpha/2)) ) or ( np.abs(bearing - th_k_sens) > (beta/2) ):
                    update_amt = l_0
                # elif (z_k_t < z_max) and ( np.abs(bearing - th_k_sens) > (beta/2) ):
                elif (z_k_t < z_max) and ( np.abs(r - z_k_t) > (alpha/2) ):
                    update_amt = l_occ
                elif r <= z_k_t:
                    update_amt = l_free

                world[y,x] += update_amt - l_0

        # animate
        # make origin of world at bottom left, not top left
        proper_orientation = np.flipud(world)
        # convert from log odds to regular probability (between 0 and 1)
        e_l = np.exp(proper_orientation)
        reg_probability = e_l / (1 + e_l)
        # show where robot is
        x_idx, y_idx = int(states[0,t]), int(states[1,t])
        y_idx = world.shape[0] - 1 - y_idx # have to index from bottom now, not top
        temp = reg_probability[y_idx, x_idx]
        reg_probability[y_idx, x_idx] = 0
        # save world image to animation stack
        im = plt.imshow(reg_probability, cmap='gray')
        ims.append([im])
        # save image to file later (remove robot from saved map)
        reg_probability[y_idx, x_idx] = temp
        mapped_world = reg_probability

        loop.update(1)
    loop.close()

    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                    repeat=False) 
    plt.pause(.1)
    input("<Hit enter to close>")

    # save the world that was mapped (the last image on the animation stack)
    np.save("./final_world", mapped_world)