from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Tuple

from hydra.core.config_store import ConfigStore


@dataclass
class DefaultConfig:
    # training params
    algo_name: str = ""
    batch_size: int = 256
    buffer_size: int = int(1e5)
    num_epochs: int = 500
    steps_per_epoch: int = 1000
    start_steps: int = 1000
    # env params
    domain_name: str = "cheetah"
    task_name: str = "run"
    num_env: int = 10
    action_repeat: int = 4
    img_size: int = 84
    camera_id: int = 0
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 10
    # general params
    train_seed: int = 0
    eval_seed: int = 42
    # exp params
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env_name: str = "dm_control/${domain_name}-${task_name}-v0"
    exp_name: str = "${algo_name}_${domain_name}-${task_name}-v0"
    run_name: str = "${timestamp}_${exp_name}_${train_seed}"
    run_dir: str = "${hydra:runtime.cwd}/logs/${exp_name}/${run_name}"

    # hydra params
    hydra: Any = field(
        default_factory=lambda: {
            "run": {"dir": "${run_dir}"},
            "sweep": {"dir": ".tmp"},
            "output_subdir": "${run_dir}",
            "job": {"name": "debug", "chdir": "True"},
        }
    )

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"cfg": "cheetah_run"},
            {"override hydra/job_logging": "colorlog"},
            {"override hydra/hydra_logging": "colorlog"},
        ]
    )


@dataclass(kw_only=True)
class DrQConfig(DefaultConfig):
    # training params
    algo_name: str = "drq"
    # model params
    latent_dim: int = 50
    hidden_dim: int = 256
    gamma: float = 0.99
    critic_tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    init_alpha: float = 0.1
    auto_alpha: bool = True
    # encoder params
    features_sizes: Tuple[int, int, int, int] = field(
        default_factory=lambda: (32, 64, 128, 256)
    )
    kernel_sizes: Tuple[int, int, int, int] = field(
        default_factory=lambda: (3, 3, 3, 3)
    )
    strides: Tuple[int, int, int, int] = field(default_factory=lambda: (2, 2, 2, 2))


cs = ConfigStore.instance()
cs.store(name="drq_config", node=DrQConfig)
