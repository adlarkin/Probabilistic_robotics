import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2
from numpy import matmul as mm
from numpy.linalg import inv as mat_inv
from scipy.io import loadmat
import pdb
from numpy.random import normal as randn
from matplotlib import animation
from matplotlib.patches import Ellipse, Wedge
from numpy.linalg import eig

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

def get_G_t(v, w, angle, dt, F_x_matrix):
    ratio = v/w
    a =  np.array([
                    [0, 0, ( -ratio*cos(angle) ) + ( ratio*cos(angle + (w*dt)) ) ],
                    [0, 0, ( -ratio*sin(angle) ) + ( ratio*sin(angle + (w*dt)) ) ],
                    [0, 0, 0]
                    ])
    a = mm(mm(np.transpose(F_x_matrix), a), F_x_matrix)
    return np.identity(a.shape[0]) + a

def get_V_t(v, w, angle, dt):
    v_0_0 = ( -sin(angle) + sin(angle + (w*dt)) ) / w
    v_0_1 = ( (v * (sin(angle) - sin(angle + (w*dt)))) / (w*w) ) + \
        ( (v * cos(angle + (w*dt)) * dt) / w )
    v_1_0 = ( cos(angle) - cos(angle + (w*dt)) ) / w
    v_1_1 = ( -(v * (cos(angle) - cos(angle + (w*dt)))) / (w*w) ) + \
        ( (v * sin(angle + (w*dt)) * dt) / w )
    return np.array([
                    [v_0_0, v_0_1],
                    [v_1_0, v_1_1],
                    [0, dt]
                    ])

def get_M_t(a_1, a_2, a_3, a_4, v, w):
    return np.array([
                    [( (a_1 * v*v) + (a_2 * w*w) ), 0],
                    [0, ( (a_3 * v*v) + (a_4 * w*w) )]
                    ])

def make_noise(cov_matrix):
    # assume distribution is zero-centered
    noisy_transition = \
        np.random.multivariate_normal(np.zeros(cov_matrix.shape[0]), cov_matrix)
    return np.reshape(noisy_transition, (-1,1))

def get_mu_bar(prev_mu, v, w, angle, dt, F_x_matrix):
    ratio = v/w
    m = np.array([
                    [(-ratio * sin(angle)) + (ratio * sin(angle + (w*dt)))],
                    [(ratio * cos(angle)) - (ratio * cos(angle + (w*dt)))],
                    [w*dt]
                ])
    return prev_mu + mm(np.transpose(F_x_matrix), m)

def wrap(angle):
    # map angle between -pi and pi
    return (angle + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    # landmarks (x and y coordinates)
    num_landmarks = 20
    world_markers = np.random.randint(low=world_bounds[0]+1, 
        high=world_bounds[1], size=(2,num_landmarks))
    lm_x = world_markers[0,:]
    lm_y = world_markers[1,:] 

    seen_landmark = {}
    for i in range(num_landmarks):
        seen_landmark[i] = False

    # pose (x,y,theta) and landmarks (x,y)
    pose_map_size = 3 + (2*num_landmarks)

    dt = .1
    total_time = 75 # seconds
    t = np.arange(0, total_time+dt, dt)
    t = np.reshape(t, (1,-1))

    # belief (poses and landmark locations)
    # first 3 rows are pose, rest of rows are landmarks
    mu = np.zeros((pose_map_size, 1))

    ########################################################################################
    ############################## DEFINE PARAMETERS HERE ##################################
    ########################################################################################
    # noise in the command velocities (translational and rotational)
    alpha_1 = .1
    alpha_2 = .01
    alpha_3 = .01
    alpha_4 = .1
    # std deviation of range and bearing sensor noise for each landmark
    std_dev_range = .1
    std_dev_bearing = .05
    # starting belief - initial condition (robot pose)
    mu[0 , 0] = 0
    mu[1 , 0] = 0
    mu[2 , 0] = np.pi / 2
    # initial uncertainty in the poses and map (landmark locations)
    # low uncertainty in initial pose, but high uncertainty in initial landmark locations
    sigma = np.identity(pose_map_size) * 5000
    sigma[0,0] = 0
    sigma[1,1] = 0
    sigma[2,2] = 0
    # sensor field of view (in degrees ... converted to radians later)
    FOV = 60
    ########################################################################################
    ########################################################################################

    # make F_x matrix
    F_x = np.zeros((3, pose_map_size))
    F_x[0:3,0:3] = np.identity(3)

    # noise free inputs (NOT ground truth)
    v_c = 1 + (.5*sin(2*np.pi*.2*t))
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

    # make new ground truth data
    # robot's initial condition is at the center of the world
    x_pos_true[0 , 0] = 0
    y_pos_true[0 , 0] = 0
    theta_true[0 , 0] = np.pi / 2
    for timestep in range(1, t.size):
        # get previous ground truth state
        prev_state = np.zeros(mu.shape)
        prev_state[0,0] = x_pos_true[0 , timestep-1]
        prev_state[1,0] = y_pos_true[0 , timestep-1]
        prev_state[2,0] = theta_true[0, timestep-1]
        theta_prev = theta_true[0 , timestep-1]

        # get next ground truth state using previous ground truth state
        # and next ground truth input
        next_state = get_mu_bar(prev_state, velocity[0,timestep], 
            omega[0,timestep], theta_prev, dt, F_x)
        x_pos_true[0,timestep] = next_state[0,0]
        y_pos_true[0,timestep] = next_state[1,0]
        theta_true[0,timestep] = next_state[2,0]

    # a record of the pose and world estimates at each timestep
    combined_state_vecs = np.zeros((mu.shape[0],t.size))
    combined_state_vecs[:,0] = mu[:,0]

    # a record of uncertainties over time
    all_sigmas = np.zeros((sigma.shape[0], sigma.shape[1], t.size))
    all_sigmas[:,:,0] = sigma

    # run EKF SLAM
    perception_bound = np.deg2rad(FOV / 2)
    for i in range(1,t.size):
        curr_v = v_c[0,i]
        curr_w = om_c[0,i]
        prev_theta = mu[2,0]

        mu_bar = get_mu_bar(mu, curr_v, curr_w, prev_theta, dt, F_x)

        G_t = get_G_t(curr_v, curr_w, prev_theta, dt, F_x)

        V_t = get_V_t(curr_v, curr_w, prev_theta, dt) 
        M_t = get_M_t(alpha_1, alpha_2, alpha_3, alpha_4, curr_v, curr_w)
        R_t = mm( mm(V_t, M_t), np.transpose(V_t) )
        sigma_bar = mm(G_t, mm(sigma, np.transpose(G_t))) + \
            mm(np.transpose(F_x), mm(R_t, F_x))

        # correction (updating belief based on landmark readings)
        real_x = x_pos_true[0 , i]
        real_y = y_pos_true[0 , i]
        real_theta = theta_true[0 , i]
        for j in range(num_landmarks):
            m_j_x = lm_x[j]
            m_j_y = lm_y[j]
            bel_x = mu_bar[0 , 0]
            bel_y = mu_bar[1 , 0]
            bel_theta = mu_bar[2 , 0]

            # get the sensor measurement
            diff_x = m_j_x - real_x
            diff_y = m_j_y - real_y
            q_true = (diff_x ** 2) + (diff_y ** 2)
            z_true = np.array([
                            [np.sqrt(q_true)],
                            [arctan2(diff_y, diff_x) - real_theta]
                            ])
            # pdb.set_trace()
            z_true += make_noise(Q_t)

            # make sure the landmark is in the field of view (bearing vs FOV)
            if np.abs(wrap(z_true[1,0])) > perception_bound:
                continue

            lm_idx = 3 + (2*j) # where we index the mu vector for the j'th landmark

            if not seen_landmark[j]:
                r = z_true[0,0]
                bearing = z_true[1,0]
                mu_bar[lm_idx, 0] = bel_x + (r * cos(bearing + bel_theta))     # lm_x_bar
                mu_bar[lm_idx + 1, 0] = bel_y + (r * sin(bearing + bel_theta)) # lm_y_bar
                
                seen_landmark[j] = True

            # diff_x, diff_y
            delta = np.array([
                                [mu_bar[lm_idx, 0] - bel_x],
                                [mu_bar[lm_idx + 1, 0] - bel_y]
                            ])
            q = mm(np.transpose(delta), delta)
            q = q[0,0]

            z_hat = np.array([
                            [np.sqrt(q)],
                            [arctan2(delta[1,0], delta[0,0]) - bel_theta]
                            ])
            
            F_x_j = np.zeros((5, pose_map_size))
            F_x_j[0:3,0:3] = np.identity(3)
            F_x_j[3:,lm_idx:lm_idx+2] = np.identity(2)

            sqrt_q = np.sqrt(q)
            d_x = delta[0,0]
            d_y = delta[1,0]
            x_sqrt = sqrt_q * d_x
            y_sqrt = sqrt_q * d_y
            a = np.array([
                            [-x_sqrt, -y_sqrt, 0, x_sqrt, y_sqrt],
                            [d_y, -d_x, -q, -d_y, d_x]
                        ])
            H_t = (1 / q) * mm(a, F_x_j)

            S_t = mm(H_t, mm(sigma_bar, np.transpose(H_t))) + Q_t
            K_t = mm(sigma_bar, mm(np.transpose(H_t), mat_inv(S_t)))
            
            z_diff = z_true - z_hat
            z_diff[1,0] = wrap(z_diff[1,0])
            mu_bar = mu_bar + mm(K_t, z_diff)
            
            sigma_bar = mm((np.identity(sigma_bar.shape[0]) - mm(K_t, H_t)), sigma_bar)

        # update belief/uncertainty and save it for later
        mu = mu_bar
        sigma = sigma_bar
        combined_state_vecs[:,i] = mu[:,0]
        all_sigmas[:,:,i] = sigma

    # make things a list (easier for plotting)
    x_pos_true = x_pos_true.tolist()[0]
    y_pos_true = y_pos_true.tolist()[0]
    theta_true = theta_true.tolist()[0]
    t = t.tolist()[0]

    animate((x_pos_true, y_pos_true, theta_true), combined_state_vecs, 
        (lm_x, lm_y), all_sigmas, FOV)

    ''' show the final covariance matrix (optional) '''
    # p2 = plt.figure(2)
    # plt.imshow(sigma)
    # plt.draw()
    # plt.pause(.1)
    # input("<Hit enter to close>")