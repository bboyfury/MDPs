import numpy as np

# Define the states and initial belief state
states = ['1', '2*', '3', '4']
initial_belief_state = np.array([0.333,  0.333,0.000, 0.333])

# Define the transition probabilities for actions EAST and WEST
success_prob = 0.9
fail_prob = 0.1

# Transition matrices for EAST and WEST
# states x 
T_EAST = np.array([
    [0.1, 0.1, 0.0, 0.0],  # From state 1: 10% chance to stay in 1, 10% chance to move to 2, no chance to move to 3 or 4
    [0.9, 0.0, 0.1, 0.0],  # From state 2: 90% chance to move back to 1, 0% chance to stay in 2, 10% chance to move to 3, no chance to move to 4
    [0.0, 0.1, 0.1, 0.9],  # From state 3: No chance to move to 1, 10% chance to move back to 2, 10% chance to stay in 3, 90% chance to move to 4
    [0.0, 0.0, 0.9, 0.9]   # From state 4: No chance to move to 1 or 2, 90% chance to move back to 3, 90% chance to stay in 4
])


T_WEST = np.array([ # I have defined this matrices based on the following assumptions:
    [0.1, 0.1, 0.0, 0.0],  # From state 1: 10% chance to stay in 1, 10% chance to move to 2, cannot move to 3 or 4
    [0.1, 0.0, 0.1, 0.9],  # From state 2: 10% chance to move back to 1, cannot stay in 2, 10% chance to move to 3, 90% chance to move to 4
    [0.9, 0.1, 0.1, 0.0],  # From state 3: 90% chance to move back to 1, 10% chance to move back to 2, 10% chance to stay in 3, cannot move to 4
    [0.9, 0.9, 0.0, 0.0]   # From state 4: 90% chance to move back to 1, 90% chance to move back to 2, cannot move to 3 or stay in 4
])



# Observation probabilities
O = np.array([1, 1, 0, 1])  # 1 if non-goal state, 0 if goal state

def update_belief_state(belief_state, action):
    if action == 'EAST':
        T = T_EAST
    elif action == 'WEST':
        T = T_WEST
    else:
        raise ValueError("Invalid action. Choose 'EAST' or 'WEST'.")

    # Update belief state
    b_prime = np.dot(T, belief_state) * O
    b_prime /= np.sum(b_prime)  # Normalize

    return b_prime

# Initial belief state
belief_state = initial_belief_state
print("initial Belief state:", np.round(belief_state, 3))



# Number of EAST actions to take
num_actions = 5
action='EAST'
# action='WEST'


for i in range(num_actions):
    belief_state = update_belief_state(belief_state, action)
    print(f"Belief state after EAST action {i+1}:", np.round(belief_state, 3))