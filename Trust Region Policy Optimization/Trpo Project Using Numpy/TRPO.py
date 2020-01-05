import autograd.numpy as np
from autograd import hessian
from autograd import grad
import math as m
import random

#Assumptions
# Assumming that 

# Global veriables
num_of_actions = 4
grid_dimension = None
nun_of_parameters = 3 # Dimension of theta (final) 
R = None # Reward maatrix
state_trans_prb_matrix = None
discount_factor = 0.999
delta = 0.000001
pi_pr = None
# Compute state transition matrix
def compute_state_trans_prb_matrix():
    n = grid_dimension
    # Up
    for i in range (0, n * n):
        if(i - n < 0) :
            state_trans_prb_matrix[0][i][i] = 1
        else:
            state_trans_prb_matrix[0][i][i - n] = 1
        total_prb_for_rest =  0
        nr_other_states = 3
        can_left = True
        can_right = True
        can_down = True
        if((i % n) == 0):
            nr_other_states -= 1
            can_left = False
        if((i % n) == n - 1):
            nr_other_states -= 1
            can_right = False
        if((i / n) >= (n - 1)):
            nr_other_states -= 1
            can_down = False
        prb_for_rest_each = total_prb_for_rest / nr_other_states
        if(can_left):
            state_trans_prb_matrix[0][i][i - 1] = prb_for_rest_each
        if(can_right):
            state_trans_prb_matrix[0][i][i + 1] = prb_for_rest_each
        if(can_down):
            state_trans_prb_matrix[0][i][i + n] = prb_for_rest_each
    # Right
    for i in range (0, n * n):
        if((i % n) == (n -1)) :
            state_trans_prb_matrix[1][i][i] = 1
        else:
            state_trans_prb_matrix[1][i][i + 1] = 1
        total_prb_for_rest = 0
        nr_other_states = 3
        can_left = True
        can_up = True
        can_down = True
        if((i % n) == 0):
            nr_other_states -= 1
            can_left = False
        if((i / n) < 1):
            nr_other_states -= 1
            can_up = False
        if((i / n) >= (n - 1)):
            nr_other_states -= 1
            can_down = False
        prb_for_rest_each = total_prb_for_rest / nr_other_states
        if(can_left):
            state_trans_prb_matrix[1][i][i - 1] = prb_for_rest_each
        if(can_up):
            state_trans_prb_matrix[1][i][i - n] = prb_for_rest_each
        if(can_down):
            state_trans_prb_matrix[1][i][i + n] = prb_for_rest_each      
    # Down
    for i in range (0, n * n):
        if(i + n >= n * n) :
            state_trans_prb_matrix[2][i][i] = 1
        else:
            state_trans_prb_matrix[2][i][i + n] = 1
        total_prb_for_rest = 0
        nr_other_states = 3
        can_left = True
        can_up = True
        can_right = True
        if((i % n) == 0):
            nr_other_states -= 1
            can_left = False
        if(i < n):
            nr_other_states -= 1
            can_up = False
        if((i % n) == n - 1):
            nr_other_states -= 1
            can_right = False
        prb_for_rest_each = total_prb_for_rest / nr_other_states
        if(can_left):
            state_trans_prb_matrix[2][i][i - 1] = prb_for_rest_each
        if(can_up):
            state_trans_prb_matrix[2][i][i - n] = prb_for_rest_each
        if(can_right):
            state_trans_prb_matrix[2][i][i + 1] = prb_for_rest_each
    # Left
    for i in range (0, n * n):
        if(i % n == 0) :
            state_trans_prb_matrix[3][i][i] = 1
        else:
            state_trans_prb_matrix[3][i][i - 1] = 1
        total_prb_for_rest = 0
        nr_other_states = 3
        can_down = True
        can_up = True
        can_right = True
        if((i / n) >= (n - 1)):
            nr_other_states -= 1
            can_down = False            
        if(i < n):
            nr_other_states -= 1
            can_up = False
        if((i % n) == n - 1):
            nr_other_states -= 1
            can_right = False
        prb_for_rest_each = total_prb_for_rest / nr_other_states
        if(can_down):
            state_trans_prb_matrix[3][i][i + n] = prb_for_rest_each
        if(can_up):
            state_trans_prb_matrix[3][i][i - n] = prb_for_rest_each
        if(can_right):
            state_trans_prb_matrix[3][i][i + 1] = prb_for_rest_each


def get_rand_number(min_value, max_value):
    range = max_value - min_value
    choice = random.uniform(0,1)
    return min_value + range*choice

def get_index_of_rand(arr):
    '''
    param: arrary of doubles  that sums up to 1
    
    Generate a random number(0-1), and do search to find the index of this
    element on the array.
    sum of (arr{0 to (rand - 1)}) < arr[rand] <= sum of (arr{0 to rand})
    '''
    rand = get_rand_number(0, 1)
    for i in range(arr.size):
        rand = rand - arr[i]
        if(rand < 0):
            return i
    return -1

def get_stationary_distribution(policy_action_matrix):
        n = grid_dimension
        global pi_pr
        for i in range(4):
            policy_column = policy_action_matrix[:, i]
            pi_pr = pi_pr + np.multiply(state_trans_prb_matrix[i], policy_column[:, np.newaxis])
        A = np.identity(n*n)
        A = pi_pr.transpose() - A
        A = np.append(A, np.ones((1, n*n)), axis=0)
        b = np.append(np.zeros((n * n, 1)), np.ones((1, 1)), axis=0)
        A_t = A.transpose()
        C = A_t @ A
        E = A_t @ b
        d = np.linalg.solve(C, E)
        return d.transpose()[0]

def single_path_simulator(pi, num_of_traj):
    '''
    Return a tuple that contains the state, action
    '''
    states = [None] * (num_of_traj + 1)
    actions = [None] *  num_of_traj
    # Generate s0
    d = get_stationary_distribution(pi)
    # print(d)
    curr_state = 0
    for i in range(num_of_traj):
        states[i] = curr_state
        # Randomly generate the action_i
        action = get_index_of_rand(pi[curr_state])
        actions[i] = action
        curr_state = np.argmax(state_trans_prb_matrix[action][curr_state])
    states[-1] = curr_state
    counts = [0] * 4
    
    return (states, actions)

# State-action reward along the trajectory
def get_Q(states):
    i = len(states) - 2
    Q = [-1] * (len(states) - 1)
    Q_future = 0
    while (i >= 0):
        s_before = states[i]
        s_after = states[i + 1]
        curr_reward = R[s_before][s_after]
        Q[i] = curr_reward + discount_factor * Q_future
        Q_future = curr_reward + discount_factor * Q_future
        i -= 1
    return Q

# Need this to (vectorization) to compute the hassin matrix, get_pi cannot work
# with auto gradient
def get_pi(theta):
    e = m.exp(1)
    policy_action_matrix = np.zeros((grid_dimension * grid_dimension, num_of_actions))
    phis = np.zeros((4, grid_dimension * grid_dimension, 3))
    # phi_ij: j is the action, and i is the
    # state
    for i in range(grid_dimension * grid_dimension):
        for j in range (num_of_actions):
            phi = np.zeros(nun_of_parameters)
            phi[0] = m.log(i / grid_dimension + 2) * 2
            phi[2] = (i + 1 - j ** 2) * m.log((j + 2))
            phi[1] = m.log(j + 2) ** (m.log(i + 2)) 
            phis[j][i] = phi
    policy_action_matrix = (e ** (phis @ theta)).transpose() ** 2
    row_sum = np.sum(policy_action_matrix, axis=1)
    policy_action_matrix = policy_action_matrix / row_sum[:, None]
    return policy_action_matrix

# DKL(pi, pi_k) = Average of (pi_k * log(pi_k / pi))
def get_kl(theta, pi_k):
    p = get_pi(theta)
    return (pi_k * (np.log(pi_k / p))).sum() / num_of_actions

def get_gradient(theta, s, a):
    pi = get_pi(theta)
    return pi[s][a]

# L = 1/(N) * sigma pi(s_i, a_i) / pi_k(si, ai) * Q_k(si, ai)
# Convert L to be L = gradient(L_k) * (s - s_k)  
def get_gradient_of_L(theta, pi_k, states, actions, Q):
    gradient_pi = np.zeros((grid_dimension * grid_dimension, num_of_actions, nun_of_parameters))
    for i in range(grid_dimension * grid_dimension):
        for j in range(num_of_actions):
            gradient_pi[i][j] = grad(get_gradient)(theta, i, j)
    # Gradient_L = 1/N sigma (gradient(pi) / pi_old)* (Q_old/)
    sample_size = len(actions)
    gradient_L = np.zeros(nun_of_parameters)
    for i in range(sample_size):
       s = states[i]
       a = actions[i]
       gradient__pi_s_a = gradient_pi[s][a]
       pi_k_s_a = pi_k[s][a]
       Q_s_a = Q[s]
       gradient_L += gradient__pi_s_a / pi_k_s_a * Q_s_a
    return gradient_L / sample_size

def get_s(g, H):
    return np.linalg.inv(H) @ g.transpose()

# beta = sqrt((2 * delta) / (s H s.t))
def get_beta(H, s):
    denominator = s @ H @ s.transpose()
    numerator = 2 * delta
    return np.sqrt(numerator / denominator)

# theta = theta_old + beta * s
def update_theta(theta_old, s, beta):
    return theta_old + beta * s

def get_state_value_vector(gamma, stationary_distribution):
    reward_matrix = R
    n = grid_dimension
    # Reward(s) matrix
    Re = np.sum(np.multiply(pi_pr, reward_matrix), axis=1)
    # I - gamma pi_pr
    c = np.identity(n * n) - gamma * pi_pr
    V = Re @ np.linalg.inv(c)
    return(V)
# Set up
fp = open('input.txt', 'r')
grid_dimension = int(fp.readline())
common_reward = int(fp.readline())
R = np.ones((grid_dimension * grid_dimension, grid_dimension * grid_dimension)) * common_reward
final_reward = int(fp.readline())
R[grid_dimension * grid_dimension - 1 - grid_dimension][grid_dimension * grid_dimension - 1] = final_reward
R[grid_dimension * grid_dimension - 1 - 1][grid_dimension * grid_dimension - 1] = final_reward
R[0][1] = 10
R[1][2] = 6
R[0][3] = -5
R[3][4] = 7
n = grid_dimension
state_trans_prb_matrix = np.zeros((num_of_actions, grid_dimension * grid_dimension, grid_dimension * grid_dimension))
compute_state_trans_prb_matrix()
v_0 = 0
v_N = 0
#-------------
theta = np.ones(nun_of_parameters) * 3
iter = 10000
for i in range(iter):
    #print(theta)
    pi = get_pi(theta)
    pi_pr = np.zeros((n * n, n * n))
    (states, actions) = single_path_simulator(pi, 100)
    #print(actions)
    Q = get_Q(states)
    #print(sum(Q))
    g = get_gradient_of_L(theta, pi, states, actions, Q)
    #print(g)
    dff = hessian(get_kl)
    H = dff(theta, pi)
    s = get_s(g, H)
    #print(s)
    beta = get_beta(H, s)
    d = get_stationary_distribution(pi)
    V = get_state_value_vector(discount_factor, d)
    #print(pi)
    if(i == 0):
        v_0 = V.sum()
    if(i == iter - 1):
        v_N = V.sum()
    print("iter " + str(i))
    print(V.sum())
    #print(V.sum())
    #print(beta)
    theta = update_theta(theta, s, beta)
print("Value at iter 0")
print(v_0)
print("\n")
print("Value at iter last iteration 10000\n" )
print(v_N)
print("\n")

    #print(theta)
##single_path_simulator(pi, 100000)
#dff = hessian(get_kl)
#print(dff(theta, pi))

# (4,4,4)


## pi[i][j] = pow(e, phi.dot(theta)), i:= state, j:= action
#def get_pi(theta):
#    e = m.exp(1)
#    policy_action_matrix = np.zeros((grid_dimension * grid_dimension, num_of_actions))
#    # phi_ij: j is the action, and i is the
#    # state
#    for i in range(grid_dimension * grid_dimension):
#        for j in range (num_of_actions):
#            phi = np.zeros(nun_of_parameters)
#            phi[0] = m.log(i / grid_dimension + 2) ** 4
#            phi[1] = (i + 1 - j ** 2) * m.log((j + 2))
#            phi[2] = m.log(j + 2) ** (m.log(i + 2))
#            policy_action_matrix[i][j] = e ** np.dot(phi,theta).sum()
#    row_sum = np.sum(policy_action_matrix, axis=1)
#    policy_action_matrix = policy_action_matrix / row_sum[:, None]
#    return policy_action_matrix
