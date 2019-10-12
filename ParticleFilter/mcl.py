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

def animate(true_states, belief_states, markers):
    x_tr, y_tr, th_tr = true_states
    x_guess, y_guess = belief_states
    
    radius = .5
    yellow = (1,1,0)
    black = 'k'
    world_bounds = [-10,10]
    
    fig = plt.figure()
    ax = plt.axes(xlim=world_bounds, ylim=world_bounds)
    ax.set_aspect('equal')
    ax.plot(markers[0], markers[1], '+', color=black)
    actual_path, = ax.plot([], [], color='b', zorder=-2, label="Actual")
    pred_path, = ax.plot([], [], color='r', zorder=-1, label="Predicted")
    heading, = ax.plot([], [], color=black)
    robot = plt.Circle((x_tr[0],y_tr[0]), radius=radius, color=yellow, ec=black)
    ax.add_artist(robot)
    ax.legend()

    def init():
        actual_path.set_data([], [])
        pred_path.set_data([], [])
        heading.set_data([], [])
        return actual_path, pred_path, heading, robot

    def animate(i):
        actual_path.set_data(x_tr[:i+1], y_tr[:i+1])
        pred_path.set_data(x_guess[:i+1], y_guess[:i+1])
        heading.set_data([x_tr[i], x_tr[i] + radius*cos(th_tr[i])], 
            [y_tr[i], y_tr[i] + radius*sin(th_tr[i])])
        robot.center = (x_tr[i],y_tr[i])
        return actual_path, pred_path, heading, robot

    anim = animation.FuncAnimation(fig, animate, init_func=init,
        frames=len(x_tr), interval=20, blit=True, repeat=False)
    
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


if __name__ == "__main__":
    dt = .1
    t = np.arange(0, 20+dt, dt)
    t = np.reshape(t, (1,-1))

    # belief (estimates from EKF)
    mu_x = np.zeros(t.shape)
    mu_y = np.zeros(t.shape)
    mu_theta = np.zeros(t.shape)   # radians

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
    # starting belief - initial condition (robot pose)
    mu_x[0 , 0] = -5 + .5
    mu_y[0 , 0] = -3 - .7
    mu_theta[0 , 0] = (np.pi / 2) - .05
    # initial uncertainty in the belief
    sigma = np.array([
                        [1, 0, 0],  # x
                        [0, 1, 0],  # y
                        [0, 0, .1]  # theta
                    ])
    num_particles = 1000
    ########################################################################################
    ########################################################################################

    # put None as first parameter so that indexing matches alpha name
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

    # uncertainty due to measurement noise
    Q_t = np.array([
                    [(std_dev_range * std_dev_range), 0],
                    [0, (std_dev_bearing * std_dev_bearing)]
                    ])

    # ground truth
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
    
    # needed for plotting covariance bounds vs values
    bound_x = [np.sqrt(sigma[0 , 0]) * 2]
    bound_y = [np.sqrt(sigma[1 , 1]) * 2]
    bound_theta = [np.sqrt(sigma[2 , 2]) * 2]
    '''
    # needed for plotting kalman gains
    K_t = None # the kalman gain matrix that gets updated with measurements
    k_r_x = []
    k_r_y = []
    k_r_theta = []
    k_b_x = []
    k_b_y = []
    k_b_theta = []
    '''

    # TODO run mcl here
    for t_step in range(1,t.size):
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
            
            next_state = sample_motion_model(u_t, state, all_alphas, dt)
            chi_bar_t[0,i] = next_state[0,0]
            chi_bar_t[1,i] = next_state[1,0]
            chi_bar_t[2,i] = next_state[2,0]

            weight = 1
            for m in range(len(lm_x)):
                z_t = z_true[m][t_step]
                weight *= measurement_model(z_t, measurement_std_devs, next_state, lm_x[m], lm_y[m])
            chi_bar_t[-1,i] = weight

        # normalize weights
        chi_bar_t[-1,:] /= np.sum(chi_bar_t[-1,:])
        # print(np.sum(chi_bar_t[-1,:]), np.amin(chi_bar_t[-1,:]), np.amax(chi_bar_t[-1,:]))
        
        plt.subplot(121)
        plt.plot(particles[0,:], particles[1,:], '.')
        plt.title("before")

        particles = low_variance_sampler(chi_bar_t)
        
        plt.subplot(122)
        plt.plot(particles[0,:], particles[1,:], '.')
        plt.title("after")
        plt.show()
        break

    '''
    # make everything a list (easier for plotting)
    x_pos_true = x_pos_true.tolist()[0]
    y_pos_true = y_pos_true.tolist()[0]
    theta_true = theta_true.tolist()[0]
    mu_x = mu_x.tolist()[0]
    mu_y = mu_y.tolist()[0]
    mu_theta = mu_theta.tolist()[0]
    t = t.tolist()[0]

    ###############################################################################
    ###############################################################################
    # show plots and animation

    # plot the states over time
    p1 = plt.figure(1)
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
    p2 = plt.figure(2)
    plt.subplot(311)
    plt.plot(t, np.array(x_pos_true) - np.array(mu_x), color='b', label="error")
    plt.plot(t, bound_x, color='r', label="uncertainty")
    plt.plot(t, [x * -1 for x in bound_x], color='r')
    plt.ylabel("x position (m)")
    plt.legend()
    plt.subplot(312)
    plt.plot(t, np.array(y_pos_true) - np.array(mu_y), color='b')
    plt.plot(t, bound_y, color='r')
    plt.plot(t, [x * -1 for x in bound_y], color='r')
    plt.ylabel("y position (m)")
    plt.subplot(313)
    plt.plot(t, np.array(theta_true) - np.array(mu_theta), color='b')
    plt.plot(t, bound_theta, color='r')
    plt.plot(t, [x * -1 for x in bound_theta], color='r')
    plt.ylabel("heading (rad)")
    plt.xlabel("time (s)")
    plt.draw()

    # plot the kalman gains
    p3 = plt.figure(3)
    plt.plot(t[1:], k_r_x, label="Range: x position")
    plt.plot(t[1:], k_r_y, label="Range: y position")
    plt.plot(t[1:], k_r_theta, label="Range: theta")
    plt.plot(t[1:], k_b_x, label="Bearing: x position")
    plt.plot(t[1:], k_b_y, label="Bearing: y position")
    plt.plot(t[1:], k_b_theta, label="Bearing: theta")
    plt.title("Kalman gains")
    plt.ylabel("Gain")
    plt.xlabel("time (s)")
    plt.legend()
    plt.draw()

    animate((x_pos_true, y_pos_true, theta_true), (mu_x, mu_y), (lm_x, lm_y))
    ###############################################################################
    ###############################################################################
    '''