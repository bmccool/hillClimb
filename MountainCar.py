import gym
env = gym.make('MountainCar-v0')
# Action Space = Discrete(3)
# Observation Space = Box(2,)
# High [0.6  0.07]
# Low [-1.2  -0.07]

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break