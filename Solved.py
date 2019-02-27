import numpy as np
import gym
import matplotlib.pyplot as plt

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()

# Define Q-learning function
def QLearning(env, learning, discount, epsilon, min_eps, episodes):
    # We are not using a neural net (Deep Q) for this problem, so we must discretize the continuous state space
    # Determine size of discretized state space
    # Here, observation space is as follows:
        # Observation Space = Box(2,)
        # High [0.6  0.07]
        # Low [-1.2  -0.07]
    # First we multiply the range of the state space by a multiple of 10 to get rid of decimals
    num_states = (env.observation_space.high - env.observation_space.low)*\
                    np.array([10, 100])
    # Then we round to the nearest whole number + 1
    num_states = np.round(num_states, 0).astype(int) + 1
    
    # num_states now looks like [19, 15], 19 bins for the first state, 15 for the second
    # TODO is it necessary to git rid of decimals, or is it only nicer for humans?
    
    # Initialize Q table
    # The Q table holds every possible state-action pair
    # The output of the Q table is the reward for that particular state (including discounted future rewards)
    # Q table output is randomized to begin
    Q = np.random.uniform(low = -1, high = 1, 
                          size = (num_states[0], num_states[1], 
                                  env.action_space.n))
    
    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []
    
    # Calculate episodic reduction in epsilon
    # Reduction is applied to the epsilon in the "epsilon-greedy" algorithm
    # Reduction of epsilon will slide the algorithm from "explore" to "exploit"
    reduction = (epsilon - min_eps)/episodes
    
    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0,0
        state = env.reset()
        
        # Discretize state into bins
        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
    
        while done != True:   
            # Render environment for last five episodes
            if i >= (episodes - 20):
                env.render()
                
            # Determine next action - epsilon greedy strategy
            # Pick a random number - if it is greater than the epsilon term, "explore" next action randomly
            #                        if it is less than the epsilon term, "exploit" using the Q table
            if np.random.random() < 1 - epsilon:
                action = np.argmax(Q[state_adj[0], state_adj[1]]) 
            else:
                action = np.random.randint(0, env.action_space.n)
                
            # Get next state and reward
            state2, reward, done, info = env.step(action) 
            
            # Discretize state2
            state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)
            
            #Allow for terminal states
            # Note, State2 being greater than 0.5 indicates the car is near the goal.  Max is 0.6
            if done and state2[0] >= 0.5:
                Q[state_adj[0], state_adj[1], action] = reward
                
            # Adjust Q value for current state
            #TODO How does this work????
            # Update Q(s1, s2, a) based on the update rule:
            # Q’(s1, s2, a) = (1 — w)*Q(s1, s2, a) + w*(r+d*Q(s1’, s2’, argmax a’ Q(s1’, s2’, a’)))
            else:
                delta = learning*(reward + 
                                 discount*np.max(Q[state2_adj[0], state2_adj[1]]) - 
                                                 Q[state_adj[0],  state_adj[1],action])
                Q[state_adj[0], state_adj[1],action] += delta
                                     
            # Update variables
            tot_reward += reward
            state_adj = state2_adj
        
        # Decay epsilon
        if epsilon > min_eps:
            epsilon -= reduction
        
        # Track rewards
        reward_list.append(tot_reward)
        
        if (i+1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            
        if (i+1) % 100 == 0:    
            print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
            
    env.close()
    
    return ave_reward_list

# Run Q-learning algorithm
rewards = QLearning(env, 0.2, 0.9, 0.8, 0, 10000)

# Plot Rewards
plt.plot(100*(np.arange(len(rewards)) + 1), rewards)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rewards.png')     
plt.close()  