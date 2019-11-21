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
    if obs[next_pos]:
        return proba * vals[start_pos]

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
    discount_factor = 1
    m_cost = -3  # cost associated with moving

    '''
    file_data = loadmat("mdp_data.mat")
    goal = np.rot90(file_data['goal'])[1:-1 , 1:-1]
     # obstacles and goal
    non_free_states = np.rot90(file_data['map'])[1:-1 , 1:-1]
    obstacles = non_free_states - goal
    world_dim = obstacles.shape[0]
    print(obstacles[:10 , :10])
    print(obstacles.shape)

    initial_pos = (28,world_dim-1-20)   # x,y
    obstacles[initial_pos[1],initial_pos[0]] = 1

    print(np.unique(non_free_states))
    '''


    non_free_states = np.zeros((5,6), dtype=np.int8)
    non_free_states[:,0] = 1
    non_free_states[0,:] = 1
    non_free_states[-1,:] = 1
    non_free_states[:,-1] = 1
    non_free_states[2,2] = 1
    non_free_states[1:1+2,-2] = 1
    goal = np.zeros(non_free_states.shape, dtype=np.int8)
    goal[1:1+2,-2] = 1
    obstacles = non_free_states - goal
    # assign values to the goal cells
    goal_vals = np.zeros(non_free_states.shape, dtype=np.float)
    goal_vals[1,-2] = 100
    goal_vals[2,-2] = -100

    # values in each cell from value iteration (initially 0 in free cells)
    state_vals = np.zeros(non_free_states.shape, dtype=np.float)
    # set the obstacles in the world to a large negative value
    state_vals[np.where(obstacles == 1)] = -5000
    # set the goal cells to their respective values
    goal_idxs = np.where(goal == 1)
    state_vals[goal_idxs] = goal_vals[goal_idxs]
    print(state_vals,"\n")

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
                policy[start] = best_action # 0=n, 1=e, 2=s, 3=w

        if np.sum(original_vals - state_vals) == 0:
            converged = True

    # ignore the wall border
    non_free_states = non_free_states[1:-1 , 1:-1]
    state_vals = state_vals[1:-1 , 1:-1]
    policy = policy[1:-1 , 1:-1]
    print(state_vals)
    print(policy)  

    '''
    plt.subplot(121)
    plt.imshow(obstacles)
    plt.title("obstacles")
    plt.subplot(122)
    plt.imshow(goal)
    plt.title('goal')
    plt.show()
    '''

    arrow_map_dx = {}
    arrow_map_dx[0] = 0
    arrow_map_dx[1] = .4
    arrow_map_dx[2] = 0
    arrow_map_dx[3] = -.4

    arrow_map_dy = {}
    arrow_map_dy[0] = -.4
    arrow_map_dy[1] = 0
    arrow_map_dy[2] = .4
    arrow_map_dy[3] = 0
    
    plt.imshow(state_vals)
    for r in range(policy.shape[0]):
        for c in range(policy.shape[1]):
            p = policy[r,c]
            if p < 0:
                continue
            plt.arrow(c, r, arrow_map_dx[p], arrow_map_dy[p], 
                head_width=0.1, head_length=0.1, fc='k', ec='k')
    plt.title("result")
    plt.show()