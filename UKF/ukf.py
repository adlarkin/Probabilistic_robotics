import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2
from numpy import matmul as mm
from numpy.linalg import inv as mat_inv
from scipy.io import loadmat
import pdb
from numpy.random import normal as randn

def get_circle(center, radius, body_color, edge_color):
    return plt.Circle(center, radius=radius, color=body_color, ec=edge_color)

def get_pose(center, theta, radius):
    x_rotation = [center[0], center[0] + radius*np.cos(theta)]
    y_rotation = [center[1], center[1] + radius*np.sin(theta)]
    return x_rotation, y_rotation

if __name__ == "__main__":
    dt = .1
    t = np.arange(0, 20+dt, dt)
    t = np.reshape(t, (1,-1))

    # belief (estimates from EKF)
    mu_x = np.zeros(t.shape)
    mu_y = np.zeros(t.shape)
    mu_theta = np.zeros(t.shape)   # radians

    # control inputs
    velocity = np.zeros(t.shape)
    omega = np.zeros(t.shape)

    # landmarks (x and y coordinates)
    lm_x = [6, -7, 6]
    lm_y = [4, 8, -4]
    assert(len(lm_x) == len(lm_y))
    num_landmarks = len(lm_x)

    # ground truth
    x_pos_true = np.zeros(t.shape)
    y_pos_true = np.zeros(t.shape)
    theta_true = np.zeros(t.shape)  # radians

    ########################################################################################
    ############################## DEFINE PARAMETERS HERE ##################################
    ########################################################################################
    use_mat_data = False
    # noise in the command velocities (translational and rotational)
    alpha_1 = .1
    alpha_2 = .01
    alpha_3 = .01
    alpha_4 = .1
    # std deviation of range and bearing sensor noise for each landmark
    std_dev_range = .1
    std_dev_bearing = .05
    # starting belief - initial condition (robot pose)
    mu_x[0 , 0] = -5
    mu_y[0 , 0] = -3
    mu_theta[0 , 0] = (np.pi / 2)
    # initial uncertainty in the belief
    sigma = np.array([
                        [.1, 0, 0],  # x
                        [0, .1, 0],  # y
                        [0, 0, .1]  # theta
                    ])
    # unscented transform params
    ut_alpha = .4
    kappa = 4
    beta = 2
    ########################################################################################
    ########################################################################################

    # uncertainty due to measurement noise
    Q_t = np.array([
                    [(std_dev_range * std_dev_range), 0],
                    [0, (std_dev_bearing * std_dev_bearing)]
                    ])

    # set ground truth data
    if use_mat_data:
        # ground truth comes from file

        # all loaded vars are numpy arrays
        # all have shape of (1, 201)
        x = loadmat('hw2_soln_data.mat')
        
        # time
        t = x['t']
        # control inputs
        omega = x['om']
        velocity = x['v']
        # true states
        x_pos_true = x['x']
        y_pos_true = x['y']
        theta_true = x['th']
    else:
        # TODO make new ground truth data
        pass

    # needed for plotting covariance bounds vs values
    bound_x = [0]
    bound_y = [0]
    bound_theta = [0]
    # needed for plotting kalman gains
    K_t = None # the kalman gain matrix that gets updated with measurements
    k_r_x = []
    k_r_y = []
    k_r_theta = []
    k_b_x = []
    k_b_y = []
    k_b_theta = []

    # run UKF
    mu = np.array([mu_x[0,0], mu_y[0,0], mu_theta[0,0]])
    mu = np.reshape(mu, (-1, 1))
    # (ut parameters)
    n = mu.size
    my_lambda = ((ut_alpha * ut_alpha) * (n + kappa)) - n
    gamma = np.sqrt(n + my_lambda)
    for t_step in range(1,t.size):
        time = t[0,t_step]
        # control inputs
        vel = 1 + (.5*cos(2*np.pi*.2*time))
        omega = -.2 + (2*cos(2*np.pi*.6*time))

        # generate augmented mean and covariance
        M_t = np.zeros((2,2))
        M_t[0,0] = (alpha_1 * vel * vel) + (alpha_2 * omega * omega)
        M_t[1,1] = (alpha_3 * vel * vel) + (alpha_4 * omega * omega)
        Q_t = np.zeros((2,2))
        Q_t[0,0] = std_dev_range * std_dev_range
        Q_t[1,1] = std_dev_bearing * std_dev_bearing
        mu_t_aug = np.zeros((7,1))
        mu_t_aug[0:3,0] = mu[:,0]
        sig_aug = np.zeros((7,7))
        sig_aug[0:3,0:3] = sigma
        sig_aug[3:5,3:5] = M_t
        sig_aug[5: ,5: ] = Q_t
        # (save dimensionality for later)
        L = mu_t_aug.shape[0]
        two_L_bound = (2*L) + 1

        # generate sigma points
        chi_aug = np.zeros((7,15))
        chi_aug[:,0] = mu_t_aug[:,0]
        mat_sq_root = gamma * np.linalg.cholesky(sig_aug)
        chi_aug[:,1:8] = mu_t_aug + mat_sq_root
        chi_aug[:,8:] = mu_t_aug - mat_sq_root

        # pass sigma points through motion model and compute gaussian statistics
        chi_bar_x = np.zeros((3,15))
        angle = mu[2,0]
        for pt in range(chi_bar_x.shape[1]):
            # (get new input based on original input and sampled inputs)
            v_new = chi_aug[3,pt] + vel
            om_new = chi_aug[4,pt] + omega
            # (save how model propogates forward with new input)
            forward_input = np.zeros((3,1))
            forward_input[0,0] = ( (-v_new/om_new) * sin(angle) ) + \
                ( (v_new/om_new) * sin(angle + (om_new*dt)) )
            forward_input[1,0] = ( (v_new/om_new) * cos(angle) ) - \
                ( (v_new/om_new) * cos(angle + (om_new*dt)) )
            forward_input[2,0] = om_new*dt
            # (save new state based on propogation and previously sampled state)
            chi_bar_x[:,pt] = chi_aug[0:3,pt] + forward_input[:,0]
        # (make mean and covariance weights)
        weights_m = np.zeros((1,15))
        weights_m[0,0] = my_lambda / (L + my_lambda)
        weights_c = np.zeros((1,15))
        weights_c[0,0] = weights_m[0,0] + (1 - (ut_alpha * ut_alpha) + beta)
        for pt in range(1,weights_m.shape[1]):
            val = 1 / ( 2 * (L + my_lambda) )
            weights_m[0,pt] = val
            weights_c[0,pt] = val
        # (get new belief)
        mu_bar = np.zeros(mu.shape)
        for pt in range(two_L_bound):
            mu_bar += weights_m[0,pt] * np.reshape(chi_bar_x[:,pt], mu_bar.shape)
        # (get new uncertainty)
        sigma_bar = np.zeros(sigma.shape)
        for pt in range(two_L_bound):
            state_diff = np.reshape(chi_bar_x[:,pt], mu_bar.shape) - mu_bar
            sigma_bar += weights_c[0,pt] * mm(state_diff, np.transpose(state_diff))

        # predict observations at sigma points and compute gaussian statistics
        for i in range(num_landmarks):
            if i > 0:
                # only doing one landmark for now
                # later, this means we are one the second landmark
                # this means we need to resample points
                break
            Z_bar_t = np.zeros((2,15))
            for pt in range(Z_bar_t.shape[1]):
                bel_x = chi_bar_x[0,pt]
                bel_y = chi_bar_x[1,pt]
                bel_theta = chi_bar_x[2,pt]
                q = ( (lm_x[i] - bel_x) ** 2 ) + ( (lm_y[i] - bel_y) ** 2 )
                Z_bar_t[0,pt] = np.sqrt(q)
                Z_bar_t[1,pt] = arctan2(lm_y[i] - bel_y, lm_x[i] - bel_x) - bel_theta
            Z_bar_t += chi_aug[-2:,:]
            z_hat = np.zeros((2,1))
            for pt in range(two_L_bound):
                z_hat += weights_m[0,pt] * np.reshape(Z_bar_t[:,pt], z_hat.shape)
            S_t = np.zeros((2,2))
            for pt in range(two_L_bound):
                meas_diff = np.reshape(Z_bar_t[:,pt], z_hat.shape) - z_hat
                S_t += weights_c[0,pt] * mm(meas_diff, np.transpose(meas_diff))
            sigma_t = np.zeros((3,2))
            for pt in range(two_L_bound):
                state_diff = np.reshape(chi_bar_x[:,pt], mu_bar.shape) - mu_bar
                meas_diff = np.reshape(Z_bar_t[:,pt], z_hat.shape) - z_hat
                sigma_t += weights_c[0,pt] * mm(state_diff, np.transpose(meas_diff))
        
            # (get the true measurement for the given landmark)
            true_x = x_pos_true[0,t_step]
            true_y = y_pos_true[0,t_step]
            true_theta = theta_true[0,t_step]
            z_true = np.zeros(z_hat.shape)
            x_diff = lm_x[i] - true_x
            y_diff = lm_y[i] - true_y
            q = ( x_diff * x_diff ) + ( y_diff * y_diff )
            z_true[0,0] = np.sqrt(q) + randn(scale=std_dev_range)
            z_true[1,0] = arctan2(y_diff, x_diff) - true_theta + randn(scale=std_dev_bearing)

            # update mean and covariance
            K_t = mm(sigma_t, mat_inv(S_t))
        break
    '''
    ###############################################################################
    ###############################################################################
    # animate and plot
    radius = .5
    yellow = (1,1,0)
    black = 'k'

    world_bounds_x = [-10,10]
    world_bounds_y = [-10,10]
    
    p1 = plt.figure(1)
    for i in range(len(x_pos_true)):
        theta = theta_true[i]
        center = (x_pos_true[i],y_pos_true[i])

        # clear the figure before plotting the next phase
        plt.clf()
        
        # get the robot pose
        body = get_circle(center, radius, yellow, black)
        orientation_x, orientation_y = \
            get_pose(center, theta, radius)
        # plot the robot pose
        plt.plot(orientation_x, orientation_y, color=black)
        axes = plt.gca()
        axes.add_patch(body)

        # plot the markers
        plt.plot(lm_x, lm_y, '+', color=black)

        # animate (keep axis limits constant and make the figure a square)
        axes.set_xlim(world_bounds_x)
        axes.set_ylim(world_bounds_y)
        axes.set_aspect('equal')
        plt.pause(.001)

    # animation is done, now plot the estimated path
    step = 2
    plt.plot(mu_x[::step], mu_y[::step], '.', color='r', label="predicted")
    plt.plot(x_pos_true[::step], y_pos_true[::step], '.', color='b', label="truth")
    plt.legend()
    p1.show()

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
    p2.show()

    # plot the uncertainty in states over time
    p3 = plt.figure(3)
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
    p3.show()

    # plot the kalman gains
    p4 = plt.figure(4)
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
    p4.show()

    # keep the plots open until user enters Ctrl+D to terminal (EOF)
    try:
        input()
    except EOFError:
        pass
    ###############################################################################
    ###############################################################################
    '''