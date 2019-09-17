# https://python-control.readthedocs.io/en/latest/
 

import numpy as np
import control
from numpy import matmul as mm # matrix multiply
from numpy.linalg import inv as mat_inv # matrix inverse
import matplotlib.pyplot as plt

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
        @type control_inputs: list
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

        # noisy inputs and noisy measurements
        self.true_measurements, self.noisy_measurements = self.get_measurements()

    def get_measurements(self):
        true_measurements = []
        noisy_measurements = []
        curr_state = np.array(self.mu)
        for c_input in self.control_inputs:
            # get the next state from the control input
            next_state = mm(self.A, curr_state) + \
                            mm(self.B, c_input) + \
                            self.make_process_noise()
            # save the ground truth measurement (no noise)
            measurement = mm(self.C, next_state)
            true_measurements.append(measurement)
            # save the noisy measurement
            noisy_measurements.append(measurement + self.make_measurement_noise())

            curr_state = next_state
        return true_measurements, noisy_measurements

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
        # kalman_gains = []

        for timestep in range(len(self.control_inputs)):
            c_input = self.control_inputs[timestep]
            z = self.noisy_measurements[timestep]

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

        #     kalman_gains.append(k)
        # return kalman_gains



# define continuous state space model
b = 20      # drag coefficient
m = 100     # mass
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

# set the initial belief
mu = np.array([[0] , [0]])
sigma = np.array([[.01 , 0] , [0 , .0001]])

# set the ground truth control inputs
control_inputs = []
t = 0
while (t < 5):
    control_inputs.append(np.array([[50]]))
    t += dt
while (t < 25):
    control_inputs.append(np.array([[0]]))
    t += dt
while (t < 30):
    control_inputs.append(np.array([[-50]]))
    t += dt
while (t < 50):
    control_inputs.append(np.array([[50]]))
    t += dt

# make the robot model
uuv = Model(sys_d.A, sys_d.B, sys_d.C, sys_d.D, R, Q, mu, sigma, control_inputs, None)

uuv.kalman_filter()