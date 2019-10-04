import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2
from numpy import matmul as mm
from numpy.linalg import inv as mat_inv
from scipy.io import loadmat
import pdb
from numpy.random import normal as randn
from matplotlib import animation

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
    # unscented transform params
    ut_alpha = .4
    kappa = 4
    beta = 2
    ########################################################################################
    ########################################################################################

    # landmarks (x and y coordinates)
    lm_x = [6, -7, 6]
    lm_y = [4, 8, -4]
    assert(len(lm_x) == len(lm_y))
    num_landmarks = len(lm_x)

    # noise free inputs (NOT ground truth)
    v_c = 1 + (.5*cos(2*np.pi*.2*t))
    om_c = -.2 + (2*cos(2*np.pi*.6*t))

    # ground truth
    x_pos_true = np.zeros(t.shape)
    y_pos_true = np.zeros(t.shape)
    theta_true = np.zeros(t.shape)  # radians

    # control inputs (truth ... with noise)
    velocity = v_c + randn(scale=np.sqrt( (alpha_1*(v_c**2)) + (alpha_2*(om_c**2)) ))
    omega = om_c + randn(scale=np.sqrt( (alpha_3*(v_c**2)) + (alpha_4*(om_c**2)) ))

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
        x = loadmat('hw3_soln_data.mat')
        
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

    # needed for plotting covariance bounds vs values
    bound_x = [np.sqrt(sigma[0 , 0]) * 2]
    bound_y = [np.sqrt(sigma[1 , 1]) * 2]
    bound_theta = [np.sqrt(sigma[2 , 2]) * 2]
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
    for t_step in range(1,t.size):
        time = t[0,t_step]
        # control inputs
        vel = v_c[0,t_step]
        omega = om_c[0,t_step]

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
        # (ut parameters)
        my_lambda = ((ut_alpha * ut_alpha) * (L + kappa)) - L
        gamma = np.sqrt(L + my_lambda)

        # generate sigma points
        chi_aug = np.zeros((7,15))
        chi_aug[:,0] = mu_t_aug[:,0]
        mat_sq_root = gamma * np.linalg.cholesky(sig_aug)
        chi_aug[:,1:8] = mu_t_aug + mat_sq_root
        chi_aug[:,8:] = mu_t_aug - mat_sq_root

        # pass sigma points through motion model and compute gaussian statistics
        chi_bar_x = np.zeros((3,15))
        for pt in range(chi_bar_x.shape[1]):
            # (get new input based on original input and sampled inputs)
            v_new = chi_aug[3,pt] + vel
            om_new = chi_aug[4,pt] + omega
            angle = chi_aug[2,pt]
            # (save how model propogates forward with new input)
            forward_input = get_fwd_propogation(v_new, om_new, angle, dt)
            # (save new state based on propogation and previously sampled state)
            chi_bar_x[:,pt] = chi_aug[0:3,pt] + forward_input[:,0]
        # (make mean and covariance weights)
        weights_m = np.zeros((1,15))
        weights_m[0,0] = my_lambda / (L + my_lambda)
        weights_c = np.zeros((1,15))
        weights_c[0,0] = weights_m[0,0] + (1 - (ut_alpha * ut_alpha) + beta)
        val = 1 / ( 2 * (L + my_lambda) )
        weights_m[0,1:] = val
        weights_c[0,1:] = val
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
                x_diff = lm_x[i] - bel_x
                y_diff = lm_y[i] - bel_y
                q = ( x_diff * x_diff ) + ( y_diff * y_diff )
                Z_bar_t[0,pt] = np.sqrt(q)
                Z_bar_t[1,pt] = arctan2(y_diff, x_diff) - bel_theta
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
            mu_bar = mu_bar + mm(K_t, z_true - z_hat)
            sigma_bar = sigma_bar - mm(K_t, mm(S_t, np.transpose(K_t)))

        # update belief
        mu = mu_bar
        sigma = sigma_bar
        mu_x[0 , t_step] = mu[0 , 0]
        mu_y[0 , t_step] = mu[1 , 0]
        mu_theta[0 , t_step] = mu[2 , 0]

        # save covariances and kalman gains for plot later
        bound_x.append(np.sqrt(sigma[0 , 0]) * 2)
        bound_y.append(np.sqrt(sigma[1 , 1]) * 2)
        bound_theta.append(np.sqrt(sigma[2 , 2]) * 2)
        k_r_x.append(K_t[0 , 0])
        k_r_y.append(K_t[1 , 0])
        k_r_theta.append(K_t[2 , 0])
        k_b_x.append(K_t[0 , 1])
        k_b_y.append(K_t[1 , 1])
        k_b_theta.append(K_t[2 , 1])

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