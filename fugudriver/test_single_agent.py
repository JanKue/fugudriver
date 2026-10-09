from pathlib import Path
import yaml

from .envs import FuguSingleAgent

if __name__ == "__main__":

    parent_path = Path(__file__).parent
    CONFIG_FILE = "envs/configs/single_agent.yaml"
    with open(parent_path / CONFIG_FILE) as cf:
        env_config = yaml.safe_load(cf)

    env = FuguSingleAgent(**env_config)

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
