import numpy as np
from scipy.io import loadmat
import pdb
import matplotlib.pyplot as plt

def move(start_pos, next_pos, vals, obs, proba):
    '''
    start_pos: tuple - starting (r,c)
    next_pos: tuple - ending (r,c)
    vals: np.array - cell value array
    obs: np.array - obstacle array
    proba: float - probability associated with the action
    '''
    # if obs[next_pos]:
    #     return proba * vals[start_pos]

    return proba * vals[next_pos]

def try_action(start, all_next_pos, vals, obs, m):
    '''
    start_pos: tuple - starting (r,c)
    all_next_pos: list of tuples - ending (r,c)
    vals: np.array - cell value array
    obs: np.array - obstacle array
    m: int - cost associated with moving
    '''
    # probabilities associated with moving as commanded
    # (straight = commanded, non_straight = left/right of commanded)
    p_straight = .8
    p_non_straight = .1
    
    v1 = move(start, all_next_pos[0], vals, obs, p_straight)
    v2 = move(start, all_next_pos[1], vals, obs, p_non_straight)
    v3 = move(start, all_next_pos[2], vals, obs, p_non_straight)

    return v1 + v2 + v3 + m


if __name__ == "__main__":
    discount_factor = .995
    m_cost = -2 # cost associated with moving
    print("discount factor:",discount_factor,"R:",m_cost)

    file_data = loadmat("mdp_data.mat")
    goal = np.rot90(file_data['goal'])[1:-1 , 1:-1]
    goal = goal.astype(np.int8)
    non_free_states = np.rot90(file_data['map'])[1:-1 , 1:-1]   # obstacles, walls, and goal
    non_free_states = non_free_states.astype(np.int8)
    obstacles = non_free_states - goal
    # define goal values
    goal_val = 100000
    # define obstacle values
    obstacle_val = -5000
    # define wall values
    wall_val = -100

    world_dim = obstacles.shape[0]
    initial_pos = (28,world_dim-20)   # x,y (c,r)

    # values in each cell from value iteration
    state_vals = np.zeros(non_free_states.shape, dtype=np.float)
    state_vals[:,:] = m_cost    # initial values
    # set the obstacle values
    state_vals[np.where(obstacles == 1)] = obstacle_val
    # set the wall values
    state_vals[0,:] = wall_val
    state_vals[:,0] = wall_val
    state_vals[-1,:] = wall_val
    state_vals[:,-1] = wall_val
    # set the goal values
    state_vals[np.where(goal == 1)] = goal_val

    policy = np.zeros(state_vals.shape, dtype=np.int8)
    policy[:,:] = -1

    converged = False
    while (not converged):
        original_vals = np.array(state_vals)

        # ignore edges since they are walls
        for r in range(1,state_vals.shape[0]-1):
            for c in range(1,state_vals.shape[1]-1):
                if non_free_states[r,c]:
                    continue
                
                start = (r,c)

                # north
                pos_list = [(r-1,c) , (r,c-1) , (r,c+1)]
                north = try_action(start, pos_list, state_vals, obstacles, m_cost)
                
                # east
                pos_list = [(r,c+1) , (r+1,c) , (r-1,c)]
                east = try_action(start, pos_list, state_vals, obstacles, m_cost)

                # south
                pos_list = [(r+1,c) , (r,c-1) , (r,c+1)]
                south = try_action(start, pos_list, state_vals, obstacles, m_cost)

                # west
                pos_list = [(r,c-1) , (r+1,c) , (r-1,c)]
                west = try_action(start, pos_list, state_vals, obstacles, m_cost)

                # find the best action and update the values and policy
                all_actions = np.array([north, east, south, west])
                best_action = np.argmax(all_actions)
                max_val = all_actions[best_action]
                state_vals[start] = discount_factor * max_val
                policy[start] = best_action

        if np.sum(original_vals - state_vals) == 0:
            converged = True

    # print(state_vals[:10,:10])

    arrow_length = .35
    arrow_map_dx = {}
    arrow_map_dx[0] = 0
    arrow_map_dx[1] = arrow_length
    arrow_map_dx[2] = 0
    arrow_map_dx[3] = -arrow_length

    arrow_map_dy = {}
    arrow_map_dy[0] = -arrow_length
    arrow_map_dy[1] = 0
    arrow_map_dy[2] = arrow_length
    arrow_map_dy[3] = 0
    
    plt.imshow(state_vals)
    for r in range(policy.shape[0]):
        for c in range(policy.shape[1]):
            p = policy[r,c]
            if p < 0:
                continue
            plt.arrow(c, r, arrow_map_dx[p], arrow_map_dy[p], 
                head_width=0.4, head_length=0.35, fc='k', ec='k')
    plt.colorbar()
    # show path for an initial condition given the policy
    r = initial_pos[1]
    c = initial_pos[0]
    plt.plot([c],[r], 'r')
    while (goal[r,c] == 0):
        next_r = r
        next_c = c
        p = policy[r,c]
        if p == 0:
            # north
            next_r = r - 1
        elif p == 1:
            # east
            next_c = c + 1
        elif p == 2:
            # south
            next_r = r + 1
        else:
            # west
            next_c = c - 1
        plt.plot([c,next_c],[r,next_r], 'r')
        r = next_r
        c = next_c
    plt.show()
