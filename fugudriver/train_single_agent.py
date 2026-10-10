from pathlib import Path
import yaml
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.logger import pretty_print

from .envs import FuguSingleAgent
from .modules import DefaultSingleAgentModule


# Set config file paths here (relative paths)
ENV_CONFIG_FILE = "envs/configs/single_agent.yaml"
MODEL_TRAIN_CONFIG_FILE = "modules/configs/single_model_training.yaml"

if __name__ == "__main__":
    # Load config files
    parent_path = Path(__file__).parent
    with open(parent_path / ENV_CONFIG_FILE, "r") as ecf:
        env_config = yaml.safe_load(ecf)
    with open(parent_path / MODEL_TRAIN_CONFIG_FILE, "r") as mcf:
        model_train_config = yaml.safe_load(mcf)

    ray.init()
    tune.register_env("FuguSingleAgent", lambda cfg: FuguSingleAgent(**cfg))

    rl_module_spec = RLModuleSpec(module_class=DefaultSingleAgentModule,
                                  model_config=model_train_config["rl_module"])

    algo_config = (PPOConfig(algo_class=PPO)
            .framework("torch")
            .environment(env="FuguSingleAgent", env_config=env_config)
            .rl_module(rl_module_spec=rl_module_spec)
            .env_runners(**model_train_config["env_runners"])
            .learners(**model_train_config["learners"])
            .training(**model_train_config["training"])
    )

    print("Building algorithm ...")
    algo = algo_config.build_algo()
    print("Start training ...")
    for _ in range(5):
        train_result = algo.train()
        print(pretty_print(train_result))
    print("Saving model to path:", algo.save_to_path())
