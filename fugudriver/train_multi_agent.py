from pathlib import Path

import ray
import yaml
from ray import tune
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.logger import pretty_print

from .envs import FuguMultiAgent, FuguSingleAgent
from .modules import DefaultSingleAgentModule

# Set config file paths here (relative paths)
ENV_CONFIG_FILE = "configs/envs/multi_agent.yaml"
MODEL_TRAIN_CONFIG_FILE = "configs/train/single_model_training.yaml"

if __name__ == "__main__":
    # Load config files
    parent_path = Path(__file__).parent
    with open(parent_path / ENV_CONFIG_FILE, "r") as ecf:
        env_config = yaml.safe_load(ecf)
    with open(parent_path / MODEL_TRAIN_CONFIG_FILE, "r") as mcf:
        model_train_config = yaml.safe_load(mcf)

    ray.init()
    tune.register_env("FuguMultiAgent", lambda cfg: FuguMultiAgent(**cfg))

    dummy_env = FuguSingleAgent(pd_config=env_config["pd_config"])
    
    single_module_spec = RLModuleSpec(
        module_class=DefaultSingleAgentModule,
        model_config=model_train_config["rl_module"],
        action_space=dummy_env.action_space,
        observation_space=dummy_env.observation_space,
    )
    POLICY_NAME = "car" # all agents are controlled by the same policy
    multi_module_spec = MultiRLModuleSpec(
        rl_module_specs={POLICY_NAME: single_module_spec},
    )

    algo_config = (PPOConfig(algo_class=PPO)
            .framework("torch")
            .environment(env="FuguMultiAgent", env_config=env_config)
            .multi_agent(policies=[POLICY_NAME], policy_mapping_fn=(lambda agent_id, episode, **kwargs: POLICY_NAME))
            .rl_module(rl_module_spec=multi_module_spec)
            .env_runners(env_to_module_connector=lambda env, spaces, device: FlattenObservations(multi_agent=True),
                         **model_train_config["env_runners"])
            .learners(**model_train_config["learners"])
            .training(**model_train_config["training"])
    )

    if model_train_config.get("run_ray_tuner"):
        tuner = tune.Tuner(
            trainable=PPO,
            param_space=algo_config,
            tune_config=tune.TuneConfig(**model_train_config["tune_config"]),
            run_config=tune.RunConfig(**model_train_config["run_config"]),
        )
        tuner.fit()
    else:
        print("Training ...")
        algo = algo_config.build_algo()
        for _ in range(2):
            train_result = algo.train()
            print(pretty_print(train_result))
        print("Saving model to path:", algo.save_to_path())
