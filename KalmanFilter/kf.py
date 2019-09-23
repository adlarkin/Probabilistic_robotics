# https://python-control.readthedocs.io/en/latest/
 

import numpy as np
import control
from numpy import matmul as mm # matrix multiply
from numpy.linalg import inv as mat_inv # matrix inverse
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pdb

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

        # plotting the covariance in position after each measurement and prediction
        self.x_cov_times = []
        self.x_cov_vals = []

    def save_iteration_data(self, model, timestep, k_gains=None):
        """
        @type model: Model
        """
        self.v_sigma_pos.append(np.sqrt(model.sigma[0 , 0]) * 2)
        self.v_sigma_neg.append(np.sqrt(model.sigma[0 , 0]) * 2 * -1)
        self.v_error.append(model.vtr[0 , timestep] - model.mu[0 , 0])
        self.x_sigma_pos.append(np.sqrt(model.sigma[1 , 1]) * 2)
        self.x_sigma_neg.append(np.sqrt(model.sigma[1 , 1]) * 2 * -1)
        self.x_error.append(model.xtr[0 , timestep] - model.mu[1 , 0])
        self.v_true.append(model.vtr[0 , timestep])
        self.x_true.append(model.xtr[0 , timestep])
        self.v_pred.append(model.mu[0 , 0])
        self.x_pred.append(model.mu[1 , 0])
        if k_gains is not None:
            self.k_v.append(k_gains[0 , 0])
            self.k_x.append(k_gains[1 , 0])

    def plot(self):
        # increase vertical spacing between subplots
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html
        # plt.subplots_adjust(hspace=.4)

        x_label_str = "Time (s)"

        # error covariance plots
        p1 = plt.figure(1)
        plt.subplot(211)
        plt.plot(self.times, self.v_sigma_neg, 'r', label="Error Covariance")
        plt.plot(self.times, self.v_sigma_pos, 'r')
        plt.plot(self.times, self.v_error, 'b', label="Velocity Estimation Error")
        plt.title("Estimation error and error covariance vs time")
        plt.ylabel("Error (velocity)")
        plt.legend()
        plt.subplot(212)
        plt.plot(self.times, self.x_sigma_neg, 'r', label="Error Covariance")
        plt.plot(self.times, self.x_sigma_pos, 'r')
        plt.plot(self.times, self.x_error, 'b', label="Position Estimation Error")
        plt.ylabel("Error (position)")
        plt.xlabel(x_label_str)
        plt.legend()
        p1.show()

        # state estimates vs ground truth plots
        p2 = plt.figure(2)
        plt.plot(self.times, self.v_true, label="True Vel")
        plt.plot(self.times, self.v_pred, label="Predicted Vel")
        plt.plot(self.times, self.x_true, label="True Pos")
        plt.plot(self.times, self.x_pred, label="Predicted Pos")
        plt.title("State estimates and true states vs time")
        plt.xlabel(x_label_str)
        plt.ylabel("Position (m) and Velocity (m / s^2)")
        plt.legend()
        p2.show()

        # kalman gain plots
        # no kalman gain at the first timestep since filtering starts at second timestep
        p3 = plt.figure(3)
        plt.plot(self.times[1:], self.k_v, label="Velocity")
        plt.plot(self.times[1:], self.k_x, label="Position")
        plt.title("Kalman gain vs time")
        plt.ylabel("Gain")
        plt.xlabel(x_label_str)
        plt.legend()
        p3.show()

        p4 = plt.figure(4)
        plt.plot(self.x_cov_times, self.x_cov_vals)
        plt.title("Position covariance between prediction and measurement update")
        plt.ylabel("Covariance")
        plt.xlabel(x_label_str)
        p4.show()

        # keep the plots open until user enters Ctrl+D to terminal (EOF)
        try:
            input()
        except EOFError:
            return


class Model():
    def __init__(self, times, A, B, C, D, R, Q, 
        mu, sigma, control_inputs, seed, load_data):
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

        self.times = times

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

        if load_data:
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
        self.Q = Q
        self.R = R
        self.control_inputs = u
        self.vtr = vtr
        self.xtr = xtr
        self.measurements = z

    def get_data(self):
        # initial condition is v=0 and x=0
        curr_state = np.array([[0] , [0]])
        # input and measurement at start at timestep 1 since KF starts at timestep 1
        vtr = np.zeros((1,self.control_inputs.size))
        xtr = np.zeros((1,self.control_inputs.size))
        measurements = np.zeros((1,self.control_inputs.size))

        # updates begin at timestep 1 since timestep 0 is the initial state
        for i in range(1, self.control_inputs.size):
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
        # for plotting
        self.plotter = Plotter(self.times)

        self.plotter.save_iteration_data(self, 0, None)
        self.plotter.x_cov_times.append(0)
        self.plotter.x_cov_vals.append(self.sigma[1 , 1])
        for timestep in range(1,self.control_inputs.size):

            c_input = self.control_inputs[0 , timestep]
            c_input = np.reshape(c_input, (1,1))
            z = self.measurements[0 , timestep]

            # prediction
            mu_bar = mm(self.A, self.mu) + mm(self.B, c_input)
            sigma_bar = mm(self.A, mm(self.sigma, np.transpose(self.A))) + self.R
            self.plotter.x_cov_times.append(self.times[timestep])
            self.plotter.x_cov_vals.append(sigma_bar[1 , 1])
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

            self.plotter.save_iteration_data(self, timestep, k)
            self.plotter.x_cov_times.append(self.times[timestep])
            self.plotter.x_cov_vals.append(sigma[1 , 1])
    
    def plot_results(self):
        self.plotter.plot()



if __name__ == "__main__":
    ########################################################################################
    ############################## DEFINE PARAMETERS HERE ##################################
    ########################################################################################
    # use the data from the matlab file, or make our own?
    use_file_data = True
    # eliminate randomness in noise generation? (For debugging)
    rand_seed = None
    # measurement covariance (default = .001)
    z_noise = .001
    # process noise associated with velocity state (default = .01)
    v_noise = .01
    # process noise associated with position state (default = .0001)
    x_noise = .0001
    # timestep (seconds)
    dt = .05
    #  initial belief (initial condition):
    # starting v and x: [[v] , [x]]
    mu = np.array([[0] , [0]])
    # starting covariance: [[v , 0] , [0 , x]]
    sigma = np.array([[.75 , 0] , [0 , .05]])
    ########################################################################################
    ########################################################################################


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
    R = np.array([[v_noise , 0] , [0 , x_noise]])
    Q = np.array([[z_noise]])

    # set the ground truth control inputs
    input_times = np.arange(0, 50+dt, dt)
    control_inputs = np.zeros((1, input_times.size))
    control_inputs[0 , 0:100] = 50
    control_inputs[0 , 100:500] = 0
    control_inputs[0 , 500:600] = -50
    control_inputs[0 , 600:] = 0

    # make the robot model
    uuv = Model(input_times, sys_d.A, sys_d.B, sys_d.C, sys_d.D, R, Q, 
        mu, sigma, control_inputs, rand_seed, use_file_data)

    uuv.kalman_filter()
    uuv.plot_results()