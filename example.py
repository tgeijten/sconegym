import gym, deprl
import sconegym

# create the sconegym env
env = gym.make("sconerun_h2190-v0")
policy = deprl.load_baseline(env)

for ep in range(100):
    if ep % 10 == 0:
        env.store_next_episode()  # Store results of every 10th episode

    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = policy(state)
        # applies action and advances environment by one step
        state, reward, done, info = env.step(action)

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
