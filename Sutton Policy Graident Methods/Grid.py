import numpy as np
import math as m
from scipy import linalg
# ------------------------------------------------------------------------------
# Policy Gradient Methods for Reinforcement Learning with Function Approximation
# 
# By Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
# AT&T Labs – Research, 180 Park Avenue, Florham Park, NJ 07932
#-------------------------------------------------------------------------------

# Model
# Grid, n by n
# Actions: 0: up, 1: left, 2: down, 3: right (each action moves 1 unit only)
# Goal: Move from (0, 0) to (n - 1, n - 1) with the highest reward

# Input:
# First line: {grid dimension}
# Second line: the reward of any regular coordinate transitions (reward for any state transition that is not mentioned in the lines below will use this value)
# Third line: the reward that moves to (n - 1, n -1)
# Fourth to last line: {(x1, y1)} {(x2, y2)} {reward from s1 to s2}, where (x1, y1) is next to (x2, y2)

# Assumptions:
# 0. x is related to row, y is related to column
# 1. If making the action would hit the wall, it will remain at the same coordinate with a reward of (a value that is 1 smaller than the smallest reward value).
# 2. The state transition probility at any coordinate is distributed 88% to the desired state, and (12% / nr of available state) to each of the rest
# 3. Each coordinate has a state number that is (n * x + y)
#-------------------------------------------------------------------------------

# Global variable
state_trans_prb_matrix = None
reward_matrix = None
pi_pr = None
n = 0
phi = None
#-------------------------------------------------------------------------------

# Compute state transition matrix
def compute_state_trans_prb_matrix():
    global n
    global state_trans_prb_matrix
    # Up
    for i in range (0, n * n):
        if(i - n < 0) :
            state_trans_prb_matrix[0][i][i] = 0.88
        else:
            state_trans_prb_matrix[0][i][i - n] = 0.88
        total_prb_for_rest = (1 - 0.88)
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
            state_trans_prb_matrix[1][i][i] = 0.88
        else:
            state_trans_prb_matrix[1][i][i + 1] = 0.88
        total_prb_for_rest = (1 - 0.88)
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
            state_trans_prb_matrix[2][i][i] = 0.88
        else:
            state_trans_prb_matrix[2][i][i + n] = 0.88
        total_prb_for_rest = (1 - 0.88)
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
            state_trans_prb_matrix[3][i][i] = 0.88
        else:
            state_trans_prb_matrix[3][i][i - 1] = 0.88
        total_prb_for_rest = (1 - 0.88)
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

def get_policy_action_matrix(theta):
    e = m.exp(1)
    global state_trans_prb_matrix
    global phi
    phi = np.zeros((4, n * n, 4))
    policy_action_matrix = np.zeros((n * n, 4))
    # phi_ij: ij means moving from state i to j is the action, and i is the
    # state
    for i in range(n * n):
        for j in range (4):
            x1 = m.log(i + 2)
            x2 = (i + 1) * m.log((j + 2))
            x3 = m.log(j + 2) / (i + 1) 
            x4 = m.log((i + 1) * 5 - j)
            phi_ij = np.array([x1, x2, x3, x4])
            phi[j][i] = phi_ij
            policy_action_matrix[i][j] = m.pow(e, phi_ij @ theta)
    row_sum = np.sum(policy_action_matrix, axis = 1)
    policy_action_matrix = policy_action_matrix / row_sum[:, None]
    return policy_action_matrix

def get_stationary_distribution(policy_action_matrix):
        global state_trans_prb_matrix
        global n
        global pi_pr
        # Matrix of policy function X state_trans_prb_matrix
        pi_pr = np.zeros((n * n, n * n))
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
        # Eigen value method.................
        #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
        Q = pi_pr
        evals, evecs = np.linalg.eig(Q.T)
        evec1 = evecs[:,np.isclose(evals, 1)]
        #Since np.isclose will return an array, we've indexed with an array
        #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
        evec1 = evec1[:,0]
        stationary = evec1 / evec1.sum()
        stationary = stationary.real
        return d

# V = R + gamma*p*V
# V(1 - gamma * p) = R
def get_state_value_vector(gamma, stationary_distribution):
    global pi_pr
    global reward_matrix
    global n
    # Reward(s) matrix
    R = np.sum(np.multiply(pi_pr, reward_matrix), axis=1)
    # I - gamma pi_pr
    c = np.identity(n * n) - gamma * pi_pr
    V = R @ np.linalg.inv(c)
    return(V)

# Q = R + gamma * Pr(s->s', a) V(s')
def get_action_state_value_matrix(gamma, stationary_distribution, state_value_vector):
    global state_trans_prb_matrix
    PV = state_trans_prb_matrix @ state_value_vector
    R = np.zeros((n*n, 4))
    for i in range(4):
        R[:, i] = np.sum(np.multiply(state_trans_prb_matrix[i], reward_matrix), axis=1)
    return R + gamma * PV.transpose()

# We will get a 4 X n*n X 4 matrix (actions X states X features)
def get_partial_derivative_fw_matrix(policy_action_matrix):
    global phi
    # Matrix of sum of pi(s,b) * phi_sb at state s, where phi_sb is a row (feature) vector with 4 elements
    pi_phi = np.zeros((n*n, 4))
    # Partial derivative matrix of fw over w
    dfw_dw = np.zeros((4, n*n, 4))
    for i in range(4):
        pi_phi = pi_phi + np.multiply(phi[i], policy_action_matrix)
    for i in range(4):
        dfw_dw[i] = phi[i] - pi_phi
    return dfw_dw

# We w to statisfy that d * pi [Q -fw] (dfw_dw) = 0
def get_fw_matrix(dfw_dw, Q, d, pi):
    d_pi = np.zeros(((n * n), 4))
    for i in range(4):
        d_pi[:, [i]] = np.multiply(pi[:, [i]], d)
    #print(np.multiply(d_pi, Q))
    d_pi_Q = np.multiply(d_pi, Q)
    #print(np.sum(dfw_dw, axis=0))
    # Get dfw_dw that sums up the value of each row
    new_dfw_dw = np.zeros((4, (n * n)))
    for i in range(4):
        new_dfw_dw[i] = np.sum(dfw_dw[i], axis=1)
    new_dfw_dw = new_dfw_dw.transpose()
    Q_df_dw = np.multiply(new_dfw_dw, d_pi_Q)
    total_sum_Q_df_dw = np.sum(np.sum(Q_df_dw))
    # dfw_dw * dfw_dw (4 X n*n X 4), (action, state, feature vector)
    dfw_dw_dfw_dw = np.zeros((4, n * n, 4))
    for i in range(4):
        dfw_dw_dfw_dw[i] = np.multiply(dfw_dw[i], new_dfw_dw[:, [i]])
    # f * df_dw matrix
    f_df_dw = np.zeros((n*n, 4))
    for i in range(4):
        f_df_dw = f_df_dw + np.multiply(dfw_dw_dfw_dw[i], d_pi[:, [i]])
    total_sum_f_df_dw = np.sum(f_df_dw, axis=0).reshape((1, 4))
    # AT A x = AT b
    # C x = D
    total_sum_f_df_dw_trans = total_sum_f_df_dw.transpose()
    C = total_sum_f_df_dw_trans @ total_sum_f_df_dw
    D = total_sum_f_df_dw_trans * total_sum_Q_df_dw
    W = np.linalg.solve(C, D)
    # f(s,a) matrix
    f = np.zeros((n*n, 4))
    for i in range(4):
        f[:, [i]] = dfw_dw[i] @ W
    return f
    

def get_new_theta(f, d, old_theta, alpha, pi, dfw_dw):
    # dpi / dtheta = dfw / dw *pi(s,a)
    dpi_dtheta = np.zeros((4, n * n, 4))
    for i in range (4):
        dpi_dtheta[i] = np.multiply(dfw_dw[i], pi[:, [i]])
    # print(dpi_dtheta)
    # dpi / dtheta * fw
    dpi_dtheta_fw = np.zeros((n * n, 4))
    for i in range (4):
        dpi_dtheta_fw = dpi_dtheta_fw + np.multiply(dfw_dw[i], f[:, [i]])
    return  0.005 * (old_theta.transpose() + np.sum(np.multiply(dpi_dtheta_fw, d), axis=0)).transpose()

def main ():
    fp = open('input.txt', 'r')
    global n
    n = int(fp.readline())
    global state_trans_prb_matrix
    global pi_pr
    global reward_matrix
    pi_pr = np.zeros((n * n, n * n))    
    state_trans_prb_matrix = np.zeros((4, n * n, n * n))
    common_reward = int(fp.readline())
    reward_matrix = np.ones((n * n, n * n)) * common_reward
    final_reward = int(fp.readline())
    reward_matrix[n * n - 1 - n ][n * n - 1] = final_reward
    reward_matrix[n * n - 1 - 1][n * n - 1] = final_reward
    line = fp.readline()
    while (line):
        curr = line.split()
        x1 = int(curr[0])
        y1 = int(curr[1])
        state_1 = x1 * n + y1 - 1
        x2 = int(curr[2])
        y2 = int(curr[3])
        state_2 = x2 * n + y2 - 1
        reward = int(curr[4])
        reward_matrix[state_1][state_2] = reward
        line = fp.readline()
    fp.close()
    compute_state_trans_prb_matrix()
    print("State_trans_prb_matrix: \n")
    print(state_trans_prb_matrix)
    
    theta = np.array([[1], [1], [-1], [-1]])
    for i in range(10000):
        policy_action_matrix = get_policy_action_matrix(theta)
        #print('\n Policy action matrix pi(s, a) = \n')
        #print(policy_action_matrix)
        d = get_stationary_distribution(policy_action_matrix)
        #print("\n Stationary distribution: \n")
        #print(d)
        V = get_state_value_vector(0.2, d)
        Q = get_action_state_value_matrix(0.2, d, V)
        #print("\n Action state value matrix: \n")
        #print(Q)
        dfw_dw = get_partial_derivative_fw_matrix(policy_action_matrix)
        #print("\n Partial Derivative of fw over w: \n")
        #print(dfw_dw)
        f = get_fw_matrix(dfw_dw, Q, d, policy_action_matrix)
        #print("\n Apporximation function matrix fw(s,a): \n")
        #print(f)
        print("\n")
        print("Sum of V \n")
        print(sum(V))
        theta = get_new_theta(f, d, theta, 0, policy_action_matrix, dfw_dw)
    
if __name__ == "__main__" :
    main()