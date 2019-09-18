# https://python-control.readthedocs.io/en/latest/
 

import numpy as np
import control
from numpy import matmul as mm # matrix multiply
from numpy.linalg import inv as mat_inv # matrix inverse
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Model():
    def __init__(self, A, B, C, D, R, Q, mu, sigma, control_inputs, seed=None):
        """
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

        # for testing/debugging
        # self.load_matlab_data()

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
        return noisy_transition

    def kalman_filter(self):
        sigma_1 = []
        sigma_2 = []

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

            sigma_1.append(np.sqrt(sigma[0 , 0]) * 2 * -1)
            sigma_2.append(np.sqrt(sigma[0 , 0]) * 2)

            # print(mu_bar.shape)
            # print(sigma_bar.shape)
            # print(k.shape)
            # print(mu.shape)
            # print(sigma.shape)
        return sigma_1, sigma_2



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
uuv = Model(sys_d.A, sys_d.B, sys_d.C, sys_d.D, R, Q, mu, sigma, control_inputs, None)

uuv.kalman_filter()

s1, s2 = uuv.kalman_filter()

plt.plot(times, s1, times, s2)
plt.show()
