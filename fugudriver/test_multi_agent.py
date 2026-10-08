from .envs import FuguMultiAgent

if __name__ == "__main__":

    pd_config = {
        "num_agents": 16,
        "episode_length": 91,
    }
    
    env = FuguMultiAgent(pd_config=pd_config)

    for episode in range(10):
        obs, infos = env.reset(seed=episode)
        done = False
        while not done:
            actions = {aid: env.get_action_space(aid).sample() for aid in env.agents}
            obs, reward, terms, truncs, infos = env.step(action_dict=actions)
            assert all(obs[aid] in env.get_observation_space(aid) for aid in env.agents)
            done = any(terms.values()) or any(truncs.values())

        print(f"Episode over. Final rewards: {reward}; final infos: {infos}")
        env.drive_env.resample_maps() # resample map after each episode

    print("Done!")
