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
from sklearn.preprocessing import normalize
import random

radius = .5
black = 'k'

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
        norm.pdf(wrap(true_z[1,0]) - wrap(z_hat[1,0]), scale=z_std_devs[1])

def low_variance_sampler(chi):
    # don't need the weights on the new particles
    new_particles = np.zeros( (chi.shape[0]-1,chi.shape[1]) )

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

def save_uncertainty(sig_x, sig_y, sig_theta, mcl_particles):
    # 95% confidence interval, so multiply std dev by 2
    sig_x.append( np.std(particles[0,:]) * 2 )
    sig_y.append( np.std(particles[1,:]) * 2 )
    sig_theta.append( np.std(particles[2,:]) * 2 )

def bound_covariance_graph():
    axes = plt.gca()
    axes.set_ylim([-1,1])

if __name__ == "__main__":
    dt = .1
    t = np.arange(0, 20+dt, dt)
    t = np.reshape(t, (1,-1))

    ########################################################################################
    ############################## DEFINE PARAMETERS HERE ##################################
    ########################################################################################
    use_mat_data = False
    np.random.seed(None)    # reproduce noise?
    # noise in the command velocities (translational and rotational)
    alpha_1 = .1
    alpha_2 = .01
    alpha_3 = .01
    alpha_4 = .1
    alpha_5 = .01
    alpha_6 = .01
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
    lm_x = [6, -7, 6]
    lm_y = [4, 8, -4]
    assert(len(lm_x) == len(lm_y))
    num_landmarks = len(lm_x)

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

    # set ground truth data
    if use_mat_data:
        # TODO read in mat file
        raise Exception("need to save .mat file data")
    else:
        # make new ground truth data

        # robot has initial condition of position (-5,-3) and 90 degree orientation
        x_pos_true[0 , 0] = -5
        y_pos_true[0 , 0] = -3
        theta_true[0 , 0] = np.pi / 2

        # create my own ground truth states and input
        for timestep in range(1, t.size):
            # get previous ground truth state
            prev_state = np.array([x_pos_true[0 , timestep-1], 
                                    y_pos_true[0 , timestep-1],
                                    theta_true[0, timestep-1]])
            prev_state = np.reshape(prev_state, (-1,1))
            theta_prev = theta_true[0 , timestep-1]

            # get next ground truth state using previous ground truth state
            # and next ground truth input
            next_state = prev_state + \
                get_fwd_propogation(velocity[0,timestep], omega[0,timestep], theta_prev, dt)
            x_pos_true[0,timestep] = next_state[0,0]
            y_pos_true[0,timestep] = next_state[1,0]
            theta_true[0,timestep] = next_state[2,0]

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
                x_pos_true[0,i], y_pos_true[0,i], theta_true[0,i])
            z[0,0] += randn(scale=std_dev_range)
            z[1,0] += randn(scale=std_dev_bearing)
            z_true[k].append(z)

    # generate particles (initially random all over the map)
    # angles are between -pi and pi (angle wrap later to maintain these bounds)
    particles = np.zeros((3,num_particles))
    particles[0:2,:] = np.random.uniform(-10, 10, size=(2,num_particles))
    particles[-1,:] = np.random.uniform(-np.pi, np.pi, size=(1,num_particles))

    # starting belief - initial condition (robot pose)
    mu_x = []
    mu_y = []
    mu_theta = []
    save_belief(mu_x, mu_y, mu_theta, particles)
    
    # needed for plotting covariance bounds vs values
    bound_x = []
    bound_y = []
    bound_theta = []
    save_uncertainty(bound_x, bound_y, bound_theta, particles)

    print("Running MCL...")
    wait_time = .00001
    p1 = plt.figure(1)
    plt.plot(particles[0,:], particles[1,:], '.')
    plot_robot( (x_pos_true[0,0],y_pos_true[0,0]) , theta_true[0,0] )
    axes = plt.gca()
    axes.set_aspect('equal')
    plt.pause(wait_time)
    for t_step in range(1,t.size):
        print("at time %.1f (s)" % (t[0,t_step]))
        # next state/evolution of particles
        # extra row for chi_bar_t is the weight of each particle
        chi_bar_t = np.zeros((particles.shape[0]+1,particles.shape[1]))
        chi_t = np.zeros(particles.shape)

        for i in range(num_particles):
            # get inputs (add noise to represent spread in particles)
            u_vel = v_c[0,t_step] + \
                randn(scale=np.sqrt( (alpha_1*(v_c[0,t_step]**2)) + (alpha_2*(om_c[0,t_step]**2)) ))
            u_om = om_c[0,t_step] + \
                randn(scale=np.sqrt( (alpha_3*(v_c[0,t_step]**2)) + (alpha_4*(om_c[0,t_step]**2)) ))
            u_t = np.reshape([u_vel, u_om], (-1,1))

            state = np.reshape(particles[:,i], (-1,1))
            
            # motion model
            next_state = sample_motion_model(u_t, state, all_alphas, dt)
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
        particles = low_variance_sampler(chi_bar_t)

        save_belief(mu_x, mu_y, mu_theta, particles)
        save_uncertainty(bound_x, bound_y, bound_theta, particles)

        plt.clf()
        plt.plot(particles[0,:], particles[1,:], '.')
        plot_robot( (x_pos_true[0,t_step],y_pos_true[0,t_step]) , theta_true[0,t_step] )
        set_graph_bounds( particles[0:2,:], (x_pos_true[0,t_step], y_pos_true[0,t_step]) )
        plt.pause(wait_time)

    # draw the real path that the robot took in comparison to the predicted path
    plt.clf()
    plt.plot(x_pos_true[0,:],y_pos_true[0,:], label="truth")
    plt.plot(mu_x, mu_y, label="predicted")
    plot_robot( (x_pos_true[0,-1],y_pos_true[0,-1]) , theta_true[0,-1] )
    plt.plot(lm_x, lm_y, '+', color=black)
    axes = plt.gca()
    axes.set_xlim([-10,10])
    axes.set_ylim([-10,10])
    axes.set_aspect('equal')
    plt.legend()
    plt.draw()

    # make everything a list (easier for plotting)
    x_pos_true = x_pos_true.tolist()[0]
    y_pos_true = y_pos_true.tolist()[0]
    theta_true = theta_true.tolist()[0]
    t = t.tolist()[0]

    ###############################################################################
    ###############################################################################
    # show plots

    # plot the states over time
    p2 = plt.figure(2)
    plt.subplot(311)
    plt.plot(t, x_pos_true, label="true")
    plt.plot(t, mu_x, label="predicted")
    plt.ylabel("x position (m)")
    plt.legend()
    plt.subplot(312)
    plt.plot(t, y_pos_true)
    plt.plot(t, mu_y)
    plt.ylabel("y position (m)")
    plt.subplot(313)
    plt.plot(t, theta_true)
    plt.plot(t, mu_theta)
    plt.ylabel("heading (rad)")
    plt.xlabel("time (s)")
    plt.draw()

    # plot the uncertainty in states over time
    p3 = plt.figure(3)
    plt.subplot(311)
    plt.plot(t, np.array(x_pos_true) - np.array(mu_x), color='b', label="error")
    plt.plot(t, bound_x, color='r', label="uncertainty")
    plt.plot(t, [x * -1 for x in bound_x], color='r')
    plt.ylabel("x position (m)")
    plt.legend()
    bound_covariance_graph()
    plt.subplot(312)
    plt.plot(t, np.array(y_pos_true) - np.array(mu_y), color='b')
    plt.plot(t, bound_y, color='r')
    plt.plot(t, [x * -1 for x in bound_y], color='r')
    plt.ylabel("y position (m)")
    bound_covariance_graph()
    plt.subplot(313)
    plt.plot(t, np.array(theta_true) - np.array(mu_theta), color='b')
    plt.plot(t, bound_theta, color='r')
    plt.plot(t, [x * -1 for x in bound_theta], color='r')
    plt.ylabel("heading (rad)")
    plt.xlabel("time (s)")
    bound_covariance_graph()
    plt.draw()

    plt.pause(.1)
    input("<Hit enter to close>")
    ###############################################################################
    ###############################################################################