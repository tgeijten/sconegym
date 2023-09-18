import gym
import sconegym

# create the sconegym env
env = gym.make("sconewalk_h0918-v0", use_delayed_sensors=True)
action = env.action_space.sample()
for ep in range(100):
    if ep % 10 == 0:
        env.store_next_episode()  # Store results of every 10th episode

    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = env.action_space.sample()
        # applies action and advances environment by one step
        next_state, reward, done, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward

        # check if done
        if done or (ep_steps >= 1000):
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f}; \
                com={env.model.com_pos()}"
            )
            break

env.close()
