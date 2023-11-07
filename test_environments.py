import gym
import sconegym

def gym_test(environment_name):
    env = gym.make(environment_name)
    
    print(f'Testing environment {environment_name}')
    for episode in range(5):
        state = env.reset()
        env.store_next_episode()

        total_reward = 0
        for step in range(1,1000):
            # samples random action
            action = env.action_space.sample()

            # applies action and advances environment by one step
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # check if done
            if done:
                break
            
        # episode finished
        print(f'Episode {episode} finished; steps={step}; reward={total_reward:0.3f}')

    # cleanup environment
    env.close()

# evaluate all environments in gym
env_names = [env_spec.id for env_spec in gym.envs.registry.all() if env_spec.id.startswith('scone')]
for env_name in env_names:
    gym_test(env_name)
