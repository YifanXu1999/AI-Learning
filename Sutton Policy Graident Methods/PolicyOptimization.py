import numpy as np
import math as m
# ------------------------------------------------------------------------------
# Policy Gradient Methods for Reinforcement Learning with Function Approximation
# 
# By Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
# AT&T Labs – Research, 180 Park Avenue, Florham Park, NJ 07932
#_______________________________________________________________________________

# Assumptions (for convinence): 

# action: move from s_1 to s_2, and it will always happen, therefore,
# pr(s1 -> s2 | a) = 1, where a is the action moves s1 to s2, and s1 connects s2
# pr(s1 -> s2 | a) = 0, if s1 doesn't s2

# The input format will be like
# n
# {s_1 s_2 reward}
# Explaination:
#     n: num of states
#     "s_1 s_s2 reward" means s_1 can jump to s_2 with this reward
#     {*} means many 

# The intial state is always at state 0

# The theta we are going to use is a size 2 vector for convinence
# Global Variables:-------------------------------------------------------------

# Probability Transition matrix
pr_trans_matrix = None
# Reward matrix
reward_matrix = None
# Policy_action_matrix at current iteration
policy_action_matrix = None
# Number of states
n = 0

#-------------------------------------------------------------------------------

# Compute the policy action matrix that records the probablity that moves
# from s_1 to s_2 under the current policy using the function
# pi(s, a, theta) = e^(theta' phi_sa) / sum of e^(theta' phi_sb) for all action
# and phi_sa (feature vetor categorizing the state-action pair [2D])
# This function gets the action distribution 
def get_policy_action_matrix(theta) :
    e = m.exp(1)
    theta_trans = theta.transpose()
    global pr_trans_matrix
    global reward_matrix
    global policy_action_matrix
    global n
    # phi_ij: ij means moving from state i to j is the action, and i is the
    # state
    for i in range(n) :
        for j in range(n) :
            phi_ij = np.array([[pr_trans_matrix[i][j] * (- (i + j) - 0.5)], [pr_trans_matrix[i][j] * ((i + 1) * 2 + j - 1.5) ]]);
            # Get policy_action_matrix[i][j]
            policy_action_matrix[i][j] = pow(e, theta_trans @ phi_ij) - 1
    # Normalize the sum of each row to be 1
    row_sum = np.sum(policy_action_matrix, axis = 1)
    policy_action_matrix = policy_action_matrix / row_sum[:, None]
    print('Policy action matrix pi(s, a) = ')
    print(policy_action_matrix)
    compute_distribution_matrix(theta)
    

# Compute Stationary distribution matrix
def compute_distribution_matrix (theta) :
    global pr_trans_matrix
    global policy_action_matrix
    global n
    pi_p = policy_action_matrix
    print(pi_p)
    # By the property of stationary distribution matrix
    # d = d * pi * p
    # d (I - pi * p) = 0
    # Need to solve for d
    # Let it to be Ax = b
    # where x = d, b = 0, and A = (I - pi * p)
    # Need to have one extra equation x1 + x2 + x3 + ... = 1
    # With the extra equation, the size of A would become (n + 1) * n, which
    # is singular, because, rank > dimension. We want to make A' to be a square
    # matrix, so we get A_trans * A * x = A_trans * b

# Q_Matrix
# Q = R + gamma * Pr * pi(s, a) * Q
# Q(I - gamma * Pr * pi) = R
# Q = R * inv(I - gamma * Pr * pi)
def compute_state_action_value_matrix(theta, gamma) :
    global pr_trans_matrix
    global policy_action_matrix
    global n
    global reward_matrix
    a = policy_action_matrix * reward_matrix
    b = np.sum(a, axis=1)
    b = b[:, None]
    c = np.identity(n)
    c = c - gamma * policy_action_matrix  
    d = np.linalg.pinv(c)
    e = d @ b
    print(a)
    print(b)
    print('c')
    print(c)
    print(d)
    print(e)
    
    #print(Q)

def main () :
    Q = np.array([[0.44, 0.28, 0,   0.28, 0,   0,   0,   0,   0],
 [0.26, 0.22, 0.26, 0,   0.26, 0,   0,   0,   0  ],
 [0,   0.28, 0.44, 0,   0,   0.28, 0,   0,   0  ],
 [0.26, 0,   0,   0.22, 0.26, 0,   0.26, 0,   0  ],
 [0,   0.25, 0,   0.25, 0,   0.25, 0,   0.25, 0  ],
 [0,   0,   0.26, 0,   0.26, 0.22, 0,   0,   0.26],
 [0,   0,   0,   0.28, 0,   0,   0.44, 0.28, 0  ],
 [0,   0,   0,   0,   0.26, 0,   0.26, 0.22, 0.26],
 [0,   0,   0,   0,   0,   0.28, 0,   0.28, 0.44]])
    
    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(Q.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    
    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]
    
    stationary = evec1 / evec1.sum()
    
    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    print(stationary)
if __name__ == "__main__" :
    main()
    
