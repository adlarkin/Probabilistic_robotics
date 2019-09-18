# https://python-control.readthedocs.io/en/latest/
 

import numpy as np
import control
from numpy import matmul as mm # matrix multiply
from numpy.linalg import inv as mat_inv # matrix inverse
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Plotter():
    def __init__(self, times):
        """
        @type times: numpy.ndarray
        """
        self.times = times

        # error covariance and estimation error plots
        self.v_sigma_pos = []
        self.v_sigma_neg = []
        self.v_error = []

        self.x_sigma_pos = []
        self.x_sigma_neg = []
        self.x_error = []

        # state estimation plots
        self.v_pred = []
        self.v_true = []

        self.x_pred = []
        self.x_true = []

        # kalman gain plots
        self.k_v = []
        self.k_x = []

    def plot(self):
        # increase vertical spacing between subplots
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
        # plt.subplots_adjust(hspace=.4)

        x_label_str = "Time (s)"

        # error covariance plots
        p1 = plt.figure(1)
        plt.subplot(211)
        plt.plot(times, self.v_sigma_neg, 'r', label="Error Covariance")
        plt.plot(times, self.v_sigma_pos, 'r')
        plt.plot(times, self.v_error, 'b', alpha=.5, label="Velocity Estimation Error")
        plt.title("Estimation error and error covariance vs time")
        plt.ylabel("Error (velocity)")
        plt.legend()
        plt.subplot(212)
        plt.plot(times, self.x_sigma_neg, 'r', label="Error Covariance")
        plt.plot(times, self.x_sigma_pos, 'r')
        plt.plot(times, self.x_error, 'b', alpha=.5, label="Position Estimation Error")
        plt.ylabel("Error (position)")
        plt.xlabel(x_label_str)
        plt.legend()
        p1.show()

        # state estimates vs ground truth plots
        true_opacity = .6
        pred_opacity = .75
        p2 = plt.figure(2)
        plt.subplot(211)
        plt.plot(times, self.v_true, alpha=true_opacity, label="True Velocity")
        plt.plot(times, self.v_pred, alpha=pred_opacity, label="Predicted Velocity")
        plt.title("State estimates and true states vs time")
        plt.ylabel("Velocity")
        plt.legend()
        plt.subplot(212)
        plt.plot(times, self.x_true, alpha=true_opacity, label="True Position")
        plt.plot(times, self.x_pred, alpha=pred_opacity, label="Predicted Prosition")
        plt.ylabel("Postion")
        plt.xlabel(x_label_str)
        plt.legend()
        p2.show()

        # kalman gain plots
        gain = "Gain"
        p3 = plt.figure(3)
        plt.subplot(211)
        plt.plot(times, self.k_v, label="Velocity")
        plt.title("Kalman gain vs time")
        plt.ylabel(gain)
        plt.legend()
        plt.subplot(212)
        plt.plot(times, self.k_x, label="Position")
        plt.ylabel(gain)
        plt.xlabel(x_label_str)
        plt.legend()
        p3.show()

        # keep the plots open until user enters Ctrl+D to terminal (EOF)
        try:
            input()
        except EOFError:
            return


class Model():
    def __init__(self, times, A, B, C, D, R, Q, mu, sigma, control_inputs, seed=None):
        """
        @type times: numpy.ndarray
        @type A: numpy.ndarray
        @type B: numpy.ndarray
        @type C: numpy.ndarray
        @type D: numpy.ndarray
        @type R: numpy.ndarray
        @type Q: numpy.ndarray
        @type mu: numpy.ndarray
        @type sigma: numpy.ndarray
        @type control_inputs: numpy.ndarray
        """
        np.random.seed(seed)    # for debugging (removes randomness from noise)

        # state space
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # noise
        self.R = R
        self.Q = Q

        # belief
        self.mu = mu
        self.sigma = sigma

        self.control_inputs = control_inputs

        # ground truth = states (v,x), measrements = output
        self.vtr, self.xtr, self.measurements = self.get_data()

        # for plotting
        self.plotter = Plotter(times)

        # for testing/debugging
        self.load_matlab_data()

    def load_matlab_data(self):
        # loading matlab data (for comparison)
        x = loadmat('hw1_soln_data.mat')
        mu0 = x['mu0']
        Q = x['Q']
        R = x['R']
        Sig0 = x['Sig0']
        t = x['t']
        u = x['u']
        vtr = x['vtr']
        xtr = x['xtr']
        z = x['z']
        
        # try on the matlab data (for comparison)
        self.mu = mu0
        self.sigma = Sig0
        self.control_inputs = u
        self.vtr = vtr
        self.xtr = xtr
        self.measurements = z

    def get_data(self):
        vtr = np.zeros((1,self.control_inputs.size))
        xtr = np.zeros((1,self.control_inputs.size))
        measurements = np.zeros((1,self.control_inputs.size))

        curr_state = np.array(self.mu)
        for i in range(self.control_inputs.size):
            c_input = self.control_inputs[0 , i]
            c_input = np.reshape(c_input, (1,1))
            # get the next state from the control input
            next_state = mm(self.A, curr_state) + \
                            mm(self.B, c_input) + \
                            self.make_process_noise()
            # save the ground truth state (v and x)
            vtr[0 , i] = next_state[0 , 0]
            xtr[0 , i] = next_state[1 , 0]
            # save the measurement
            measurements[0 , i] = \
                (mm(self.C, next_state) + self.make_measurement_noise())

            curr_state = next_state
        return vtr, xtr, measurements

    def make_process_noise(self):
        # assume distribution is zero-centered
        noisy_transition = \
            np.random.multivariate_normal(np.zeros(self.R.shape[0]), self.R)
        return np.reshape(noisy_transition, (-1,1))

    def make_measurement_noise(self):
        # assume distribution is zero-centered
        noisy_transition = \
            np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)
        return np.reshape(noisy_transition, (-1,1))

    def kalman_filter(self):
        for timestep in range(self.control_inputs.size):
            c_input = self.control_inputs[0 , timestep]
            c_input = np.reshape(c_input, (1,1))
            z = self.measurements[0 , timestep]

            # prediction
            mu_bar = mm(self.A, self.mu) + mm(self.B, c_input)
            sigma_bar = mm(self.A, mm(self.sigma, np.transpose(self.A))) + self.R
            
            # correction
            c_transpose = np.transpose(self.C)
            matrix_one = mm(sigma_bar, c_transpose)
            matrix_two = mat_inv(mm(self.C, mm(sigma_bar, c_transpose)) + self.Q)
            k = mm(matrix_one, matrix_two)
            mu = mu_bar + mm(k, z - mm(self.C, mu_bar))
            sigma = mm(np.identity(k.shape[0]) - mm(k, self.C), sigma_bar)

            # update the model's belief for the next filter iteration
            self.mu = mu
            self.sigma = sigma

            # save info for plotting
            # error covariance and estimation error plots
            self.plotter.v_sigma_pos.append(np.sqrt(sigma[0 , 0]) * 2)
            self.plotter.v_sigma_neg.append(np.sqrt(sigma[0 , 0]) * 2 * -1)
            self.plotter.v_error.append(self.vtr[0 , timestep] - mu[0 , 0])
            self.plotter.x_sigma_pos.append(np.sqrt(sigma[1 , 1]) * 2)
            self.plotter.x_sigma_neg.append(np.sqrt(sigma[1 , 1]) * 2 * -1)
            self.plotter.x_error.append(self.xtr[0 , timestep] - mu[1 , 0])
            # state estimation plots
            self.plotter.v_pred.append(mu[0 , 0])
            self.plotter.v_true.append(self.vtr[0 , timestep])
            self.plotter.x_pred.append(mu[1 , 0])
            self.plotter.x_true.append(self.xtr[0 , timestep])
            # kalman gain plots
            self.plotter.k_v.append(k[0 , 0])
            self.plotter.k_x.append(k[1 , 0])

            # print(mu_bar.shape)
            # print(sigma_bar.shape)
            # print(k.shape)
            # print(mu.shape)
            # print(sigma.shape)
    
    def plot_results(self):
        self.plotter.plot()



# robot model parameters
b = 20      # drag coefficient
m = 100     # mass

# define continuous state space model
A = np.array([[(-b/m) , 0] , [1 , 0]])
B = np.array([[(1/m)] , [0]])
C = np.array([0, 1])
D = np.array([0])
dt = .05

# get a discrete state space model
sys = control.ss(A, B, C, D)
sys_d = control.c2d(sys, dt)
sys_d.A = np.array(sys_d.A)
sys_d.B = np.array(sys_d.B)
sys_d.C = np.array(sys_d.C)
sys_d.D = np.array(sys_d.D)

# get the system/measurement noise
R = np.array([[.01 , 0] , [0 , .0001]])
Q = np.array([[.001]])

# set the initial belief (initial condition)
mu = np.array([[0] , [0]])
sigma = np.array([[.01 , 0] , [0 , .0001]])

# set the ground truth control inputs
times = np.arange(0, 50+dt, dt)
control_inputs = np.zeros((1, times.size))
control_inputs[0 , 0:100] = 50
control_inputs[0 , 100:500] = 0
control_inputs[0 , 500:600] = -50
control_inputs[0 , 600:] = 0

# make the robot model
rand_seed = None
uuv = Model(times, sys_d.A, sys_d.B, sys_d.C, sys_d.D, R, Q, 
    mu, sigma, control_inputs, rand_seed)

uuv.kalman_filter()
uuv.plot_results()