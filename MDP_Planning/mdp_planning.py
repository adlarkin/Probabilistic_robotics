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

    # # probabilities associated with moving as commanded
    # # (straight = commanded, non_straight = left/right of commanded)
    # p_straight = .8
    # p_non_straight = .1

    my_dtype = np.int32

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


    non_free_states = np.zeros((5,6), dtype=my_dtype)
    non_free_states[:,0] = 1
    non_free_states[0,:] = 1
    non_free_states[-1,:] = 1
    non_free_states[:,-1] = 1
    non_free_states[2,2] = 1
    non_free_states[1:1+2,-2] = 1
    goal = np.zeros(non_free_states.shape, dtype=non_free_states.dtype)
    goal[1:1+2,-2] = 1
    obstacles = non_free_states - goal
    # assign values to the goal cells
    goal_vals = np.zeros(non_free_states.shape, dtype=non_free_states.dtype)
    goal_vals[1,-2] = 100
    goal_vals[2,-2] = -100

    # values in each cell from value iteration (initially 0 in free cells)
    state_vals = np.zeros(non_free_states.shape, dtype=non_free_states.dtype)
    # set the obstacles in the world to a large negative value
    state_vals[np.where(obstacles == 1)] = -5000
    # set the goal cells to their respective values
    goal_idxs = np.where(goal == 1)
    state_vals[goal_idxs] = goal_vals[goal_idxs]
    print(state_vals)

    for r in range(0,state_vals.shape[0]):
        for c in range(0,state_vals.shape[1]):
            if non_free_states[r,c]:
                continue

            if not ((r == 1) and (c == 3)):
                continue
            
            start = (r,c)

            # north
            pos_list = [(r-1,c) , (r,c-1) , (r,c+1)]
            north = try_action(start, pos_list, state_vals, obstacles, m_cost)
            
            # south
            pos_list = [(r,c+1) , (r+1,c) , (r-1,c)]
            south = try_action(start, pos_list, state_vals, obstacles, m_cost)

            # east
            pos_list = [(r+1,c) , (r,c-1) , (r,c+1)]
            east = try_action(start, pos_list, state_vals, obstacles, m_cost)

            # west
            pos_list = [(r,c-1) , (r+1,c) , (r-1,c)]
            west = try_action(start, pos_list, state_vals, obstacles, m_cost)

            all_actions = np.array([north, south, east, west])
            best_action = np.argmax(all_actions)
            max_val = all_actions[best_action]
            state_vals[start] = max_val
            print(max_val, best_action)


    plt.subplot(121)
    plt.imshow(obstacles)
    plt.title("obstacles")
    plt.subplot(122)
    plt.imshow(goal)
    plt.title('goal')
    plt.show()