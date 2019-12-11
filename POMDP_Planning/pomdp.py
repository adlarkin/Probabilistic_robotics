'''
    NOTE this solution may need to be verified for accuracy
    table 15.1 of probabilistic robotics textbook, pg 529
'''

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    T = 2       # time horizon
    gamma = 1   # discount factor

    Y = np.zeros((1,3))

    # dimension space of problem
    N = 2   # number of states
    Nu = 3  # number of control inputs
    Nz = 2  # number of measurements

    # rewards (row is state, col is action)
    r = np.array([
                    [-100, +100, -1],
                    [+100, -50,  -1]
                ])
    # transition probabilities (row is start state, col is new state)
    # 3d array, probs of u1 and u2 = 0 b/c you don't go to another state
    pt = np.zeros((N,N,Nu))
    pt[:,:,0] = 0
    pt[:,:,1] = 0
    pt[:,:,2] = np.array([
                            [.2, .8],
                            [.8, .2],
                        ])

    # measurement probabilities (row is measurement, col is state)
    pz = np.array([
                    [.7, .3],
                    [.3, .7]
                ])

    for tau in range(T):
        Ypr = None
        K = Y.shape[0]  # number of linear constraint functions
        v = np.zeros((K,Nu,Nz,N))
        for k in range(K):
            for iu in range(Nu):
                for iz in range(Nz):
                    for j in range(N):
                        # caculate v 
                        # (sum from i=1..N)
                        # +1 b/c ignoring 1st thing in row of Y, whic is u (just want v vals)
                        v[k,iu,iz,j] += Y[k,0+1] * pz[iz,0] * pt[j,0,iu]
                        v[k,iu,iz,j] += Y[k,1+1] * pz[iz,1] * pt[j,1,iu]
                            
        
        for iu in range(Nu):
            for k1 in range(K):
                for k2 in range(K):
                    v_pr = np.zeros((1,3))
                    v_pr[0,0] = iu
                    for i in range(N):
                        v_pr[0,i+1] = gamma * (r[i,iu] + v[k1,iu,0,i] + v[k2,iu,1,i])
                    if Ypr is None:
                        Ypr = np.array(v_pr)
                    else:
                        Ypr = np.vstack((Ypr, v_pr))
        # TODO prune here?
        Y = Ypr

    # remove duplicate answers
    new_array = [tuple(row) for row in Y]
    Y = np.unique(new_array, axis=0)

    # plot my results
    for line in range(Y.shape[0]):
        # only plotting solutions that have reward bounds above 25
        # (this is a hardcoded solution for pruning when T=2)
        if Y[line,1] < 25 and Y[line,2] < 25:
            continue
        plt.plot([0,1], [Y[line,2],Y[line,1]], color='k', alpha=0.6)
    # show true solution
    if T == 2:
        plt.plot([0,1],[100,-100],'--r')
        plt.plot([0,1],[42,51],'--r')
        plt.plot([0,1],[-50,100],'--r')
        plt.plot([0,1],[-61,59],'--b')
        plt.plot([0,1],[69,-21],'--b')
    elif T == 1:
        plt.plot([0,1],[100,-100],'--r')
        plt.plot([0,1],[55,40],'--r')
        plt.plot([0,1],[-50,100],'--r')

    plt.show()