import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2
from numpy import matmul as mm
from numpy.linalg import inv as mat_inv
from scipy.io import loadmat
import pdb
from numpy.random import normal as randn
from matplotlib import animation
from scipy.stats import norm
import random

radius = .5
black = 'k'

world_bounds = [-15,20]

# can look at patch collection for a cleaner, more efficient solution
# https://stackoverflow.com/questions/45969740/python-matplotlib-patchcollection-animation-doesnt-update
def animate(true_states, belief_states, markers, uncertanties, fov):
    x_tr, y_tr, th_tr = true_states
    fov_bound = np.deg2rad(fov)/2
    
    radius = .5
    yellow = (1,1,0)
    black = 'k'
    
    fig = plt.figure()
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')
    ax.plot(markers[0], markers[1], '+', color=black, zorder=-2, label="True Landmarks")
    actual_path, = ax.plot([], [], color='b', zorder=-2, label="True Path")
    pred_path, = ax.plot([], [], color='r', zorder=-1, label="Predicted Path")
    heading, = ax.plot([], [], color=black)
    robot = plt.Circle((x_tr[0],y_tr[0]), radius=radius, color=yellow, ec=black)
    ax.add_artist(robot)
    lm_uncertanties = []
    for i in range(len(markers[0])):
        next_lm_unceratinty, = ax.plot([], [], color='g')
        lm_uncertanties.append(next_lm_unceratinty)
    vision_beam = Wedge((x_tr[0],y_tr[0]), 1000, np.rad2deg(th_tr[0] - fov_bound), 
        np.rad2deg(th_tr[0] + fov_bound), zorder=-5, alpha=.1, color='m')
    ax.add_artist(vision_beam)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    def init():
        actual_path.set_data([], [])
        pred_path.set_data([], [])
        heading.set_data([], [])
        return (actual_path, pred_path, heading, robot, vision_beam) \
            + tuple(lm_uncertanties)

    def animate(i):
        for j in range(len(lm_uncertanties)):
            lm_idx = 3 + (2*j)
            x_lm = belief_states[lm_idx,i]
            y_lm = belief_states[lm_idx+1,i]
            if (x_lm == 0) and (y_lm == 0):
                # haven't seen landmark yet
                continue
            # http://anuncommonlab.com/articles/how-kalman-filters-work/part3.html#ellipses
            U, S, _ = np.linalg.svd(uncertanties[lm_idx:lm_idx+2, lm_idx:lm_idx+2, i])
            C = U * 2*np.sqrt(S)
            theta = np.linspace(0, 2*np.pi, 100)
            circle = np.array([cos(theta),sin(theta)])
            e = mm(C, circle)
            e[0,:] += x_lm
            e[1,:] += y_lm
            lm_uncertanties[j].set_data(e[0,:], e[1,:])
        actual_path.set_data(x_tr[:i+1], y_tr[:i+1])
        pred_path.set_data(belief_states[0,:i+1], belief_states[1,:i+1])
        heading.set_data([x_tr[i], x_tr[i] + radius*cos(th_tr[i])], 
            [y_tr[i], y_tr[i] + radius*sin(th_tr[i])])
        robot.center = (x_tr[i],y_tr[i])
        vision_beam.set_center((x_tr[i],y_tr[i]))
        vision_beam.theta1 = np.rad2deg(th_tr[i] - fov_bound)
        vision_beam.theta2 = np.rad2deg(th_tr[i] + fov_bound)
        return (actual_path, pred_path, heading, robot, vision_beam) \
            + tuple(lm_uncertanties)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
        frames=len(x_tr), interval=40, blit=True, repeat=False)
    
    plt.pause(.1)
    input("<Hit enter to close>")

def get_fwd_propogation(vel, om, th, delta_t):
    move_forward = np.zeros((3,1))
    ratio = vel/om
    move_forward[0,0] = ( -ratio * sin(th) ) + \
        ( ratio * sin(th + (om*delta_t)) )
    move_forward[1,0] = ( ratio * cos(th) ) - \
        ( ratio * cos(th + (om*delta_t)) )
    move_forward[2,0] = om*delta_t
    return move_forward

def get_measurement(x_marker, y_marker, x_robot, y_robot, theta_robot):
    z = np.zeros((2,1))  # [[range], [bearing]]
    x_diff = x_marker - x_robot
    y_diff = y_marker - y_robot
    q = (x_diff * x_diff) + (y_diff * y_diff)
    z[0,0] = np.sqrt(q)
    z[1,0] = arctan2(y_diff, x_diff) - theta_robot
    return z

def sample_motion_model(curr_input, x_prev, alphas, delta_t):
    assert(len(alphas) == 7)
    assert(alphas[0] is None)
    v = curr_input[0,0]
    w = curr_input[1,0]
    
    v_hat = v + randn( scale=np.sqrt( (alphas[1]*(v**2)) + (alphas[2]*(w**2)) ) )
    w_hat = w + randn( scale=np.sqrt( (alphas[3]*(v**2)) + (alphas[4]*(w**2)) ) )
    gamma = randn( scale=np.sqrt( (alphas[5]*(v**2)) + (alphas[6]*(w**2)) ) )

    x_next = get_fwd_propogation(v_hat, w_hat, x_prev[2,0], delta_t)
    x_next[0,0] += x_prev[0,0]
    x_next[1,0] += x_prev[1,0]
    x_next[2,0] += x_prev[2,0] + (gamma * delta_t)

    return x_next

def wrap(angle):
    # map angle between -pi and pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

def measurement_model(true_z, z_std_devs, predicted_state, marker_x, marker_y):
    # z_std_devs = (std_dev_range, std_dev_bearing)
    assert (len(z_std_devs) == 2)

    z_hat = get_measurement(marker_x, marker_y, predicted_state[0,0],
        predicted_state[1,0], predicted_state[2,0])

    return norm.pdf(true_z[0,0] - z_hat[0,0], scale=z_std_devs[0]) * \
        norm.pdf(wrap(true_z[1,0] - z_hat[1,0]), scale=z_std_devs[1])

def low_variance_sampler(chi):
    # don't need the weights on the new particles
    new_particles = np.zeros( (chi.shape[0]-1,chi.shape[1]) )

    saved_particle_indices = []

    M = chi.shape[1]
    r = random.uniform(0, 1/M)
    c = chi[-1,0]   # the first weight
    i = 0
    for m in range(M):
        U = r + m * (1/M)
        while U > c:
            i += 1
            c += chi[-1,i]
        new_particles[:,m] = chi[:-1,i]
        saved_particle_indices.append(i)

    # dealing with particle deprivation (not in the original algorithm)
    P = np.cov(chi[:-1,:])
    uniq = np.unique(saved_particle_indices).size   # num. of unique particles in resampling
    if (uniq/M) < .025:   # if we don't have much variety in our resampling
        Q = P / ((M*uniq) ** (1/new_particles.shape[0]))
        new_particles += mm(Q, randn(size=new_particles.shape))

    return new_particles

def plot_robot(center, heading):
    yellow = (1,1,0)

    x = center[0]
    y = center[1]

    robot = plt.Circle((x,y), radius=radius, color=yellow, ec=black)
    plt.plot([x, x + radius*cos(heading)], 
            [y, y + radius*sin(heading)], color=black)
    axes = plt.gca()
    axes.add_patch(robot)

def set_graph_bounds(particle_positions, true_position, equal_aspect=True):
    x = true_position[0]
    y = true_position[1]

    padding = .1
    max_x_particle = np.max(particle_positions[0,:]) + padding
    min_x_particle = np.min(particle_positions[0,:]) - padding
    max_y_particle = np.max(particle_positions[1,:]) + padding
    min_y_particle = np.min(particle_positions[1,:]) - padding

    padding = 1.5 * radius
    max_x_graph = max(max_x_particle, x + padding)
    min_x_graph = min(min_x_particle, x - padding)
    max_y_graph = max(max_y_particle, y + padding)
    min_y_graph = min(min_y_particle, y - padding)

    axes = plt.gca()
    axes.set_xlim([min_x_graph,max_x_graph])
    axes.set_ylim([min_y_graph,max_y_graph])

    if equal_aspect:
        axes.set_aspect('equal')

def save_belief(bel_x, bel_y, bel_theta, mcl_particles):
    bel_x.append( np.mean(mcl_particles[0,:]) )
    bel_y.append( np.mean(mcl_particles[1,:]) )
    bel_theta.append( np.mean(mcl_particles[2,:]) )

if __name__ == "__main__":
    dt = .1
    t = np.arange(0, 20+dt, dt)

    ########################################################################################
    ############################## DEFINE PARAMETERS HERE ##################################
    ########################################################################################
    np.random.seed(None)    # reproduce noise?
    # noise in the command velocities (translational and rotational)
    alpha_1 = .1 / 2
    alpha_2 = .01 / 2
    alpha_3 = .01 / 2
    alpha_4 = .1 / 2
    alpha_5 = .01 / 2
    alpha_6 = .01 / 2
    # std deviation of range and bearing sensor noise for each landmark
    std_dev_range = .1
    std_dev_bearing = .05
    ########################################################################################
    ########################################################################################

    # particles for MCL
    num_particles = int(input("Number of particles: "))

    # (put None as first parameter so that indexing matches alpha name)
    all_alphas = (None, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6)

    measurement_std_devs = (std_dev_range, std_dev_bearing)

    # landmarks (x and y coordinates)
    num_landmarks = 5
    world_markers = np.random.randint(low=world_bounds[0]+1, 
        high=world_bounds[1], size=(2,num_landmarks))
    lm_x = world_markers[0,:]
    lm_y = world_markers[1,:]

    # control inputs - noise free (NOT ground truth)
    v_c = 1 + (.5*cos(2*np.pi*.2*t))
    om_c = -.2 + (2*cos(2*np.pi*.6*t))

    # ground truth inputs (with noise)
    velocity = v_c + randn(scale=np.sqrt( (alpha_1*(v_c**2)) + (alpha_2*(om_c**2)) ))
    omega = om_c + randn(scale=np.sqrt( (alpha_3*(v_c**2)) + (alpha_4*(om_c**2)) ))

    # ground truth states
    x_pos_true = np.zeros(t.shape)
    y_pos_true = np.zeros(t.shape)
    theta_true = np.zeros(t.shape)  # radians

    ''' make new ground truth data '''
    # robot has initial condition of position (0,0) and 90 degree orientation
    x_pos_true[0] = 0
    y_pos_true[0] = 0
    theta_true[0] = np.pi / 2
    for timestep in range(1, t.size):
        # get previous ground truth state
        prev_state = np.array([x_pos_true[timestep-1], 
                                y_pos_true[timestep-1],
                                theta_true[timestep-1]])
        prev_state = np.reshape(prev_state, (-1,1))
        theta_prev = theta_true[timestep-1]

        # get next ground truth state using previous ground truth state
        # and next ground truth input
        next_state = prev_state + \
            get_fwd_propogation(velocity[timestep], omega[timestep], theta_prev, dt)
        x_pos_true[timestep] = next_state[0,0]
        y_pos_true[timestep] = next_state[1,0]
        theta_true[timestep] = next_state[2,0]

    # get true measurements
    # doing this once now to avoid redundant computation for each particle
    # we also want to make sure noise is consistent for each measurement
    # key is the landmark, which maps to a list of true measurements
    # (the i'th measurement being the true measurement for the i'th true state)
    z_true = {}
    for k in range(len(lm_x)):
        z_true[k] = []
    for i in range(t.size):
        for k in range(len(lm_x)):
            z = get_measurement(lm_x[k], lm_y[k], 
                x_pos_true[i], y_pos_true[i], theta_true[i])
            z[0,0] += randn(scale=std_dev_range)
            z[1,0] += randn(scale=std_dev_bearing)
            z_true[k].append(z)

    # generate particles (initialize them to initial pose since we know it)
    particle_poses = np.zeros((3,num_particles,t.size))
    particle_poses[0,:,0] = x_pos_true[0]
    particle_poses[1,:,0] = y_pos_true[0]
    particle_poses[2,:,0] = theta_true[0]

    # starting belief - initial condition (robot pose)
    mu_x = []
    mu_y = []
    mu_theta = []
    save_belief(mu_x, mu_y, mu_theta, particle_poses[:,:,0])

    print("Running MCL...")
    for t_step in range(1,t.size):
        print("at time %.1f (s)" % (t[t_step]))
        # next state/evolution of particles
        # extra row for chi_bar_t is the weight of each particle
        chi_bar_t = np.zeros((particle_poses.shape[0]+1,particle_poses.shape[1]))
        chi_t = np.zeros((particle_poses.shape[0], particle_poses.shape[1]))

        for i in range(num_particles):
            # get inputs (add noise to represent spread in particles)
            u_vel = v_c[t_step] + \
                randn(scale=np.sqrt( (alpha_1*(v_c[t_step]**2)) + (alpha_2*(om_c[t_step]**2)) ))
            u_om = om_c[t_step] + \
                randn(scale=np.sqrt( (alpha_3*(v_c[t_step]**2)) + (alpha_4*(om_c[t_step]**2)) ))
            u_t = np.reshape([u_vel, u_om], (-1,1))

            prev_state = np.reshape(particle_poses[:,i,t_step-1], (-1,1))
            
            # motion model
            next_state = sample_motion_model(u_t, prev_state, all_alphas, dt)
            chi_bar_t[0,i] = next_state[0,0]
            chi_bar_t[1,i] = next_state[1,0]
            chi_bar_t[2,i] = next_state[2,0]

            # measurement model
            weight = 1
            for m in range(len(lm_x)):
                z_t = z_true[m][t_step]
                weight *= measurement_model(z_t, measurement_std_devs, next_state, lm_x[m], lm_y[m])
            chi_bar_t[-1,i] = weight

        # normalize weights
        chi_bar_t[-1,:] /= np.sum(chi_bar_t[-1,:])

        # resample, factoring in the weights
        particle_poses[:,:,t_step] = low_variance_sampler(chi_bar_t)

        save_belief(mu_x, mu_y, mu_theta, particle_poses[:,:,t_step])