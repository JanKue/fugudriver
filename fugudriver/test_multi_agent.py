from pathlib import Path
import yaml

from .envs import FuguMultiAgent

if __name__ == "__main__":

    parent_path = Path(__file__).parent
    CONFIG_FILE = "envs/configs/multi_agent.yaml"
    with open(parent_path / CONFIG_FILE) as cf:
        env_config = yaml.safe_load(cf)
    
    env = FuguMultiAgent(**env_config)

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
