import gym  

env = gym.make('MountainCar-v0',render_mode="human")
for episode in range(10):
    env.reset() 
    print("Episode finished after {} timesteps".format(episode))
    for _ in range(500): 
        env.render()
#         observation, reward, down, info = env.step(env.action_space.sample())
        env.step(env.action_space.sample())
env.close()