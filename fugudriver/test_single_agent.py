from .envs import FuguSingleAgent

if __name__ == "__main__":

    pd_config = {
        "num_agents": 1,
        "episode_length": 91,
    }

    env = FuguSingleAgent(pd_config=pd_config)

    for episode in range(10):
        obs, info = env.reset(seed=episode)
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert obs in env.observation_space
            done = term or trunc

        print(f"Episode over. Final reward: {reward}; final info: {info}")
        env.drive_env.resample_maps() # resample map after each episode

    print("Done!")
