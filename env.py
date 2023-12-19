from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
from einops import rearrange
from gymnasium.wrappers import (
    ClipAction,
    FrameStackObservation,
    MaxAndSkipObservation,
    RecordVideo,
    RenderObservation,
    TransformObservation,
)
from gymnasium.wrappers.vector import (
    RecordEpisodeStatistics as VectorRecordEpisodeStatistics,
)
from shimmy.registration import register_gymnasium_envs


def make_vector_env(
    env_name: str,
    seed: int,
    num_env: int,
    img_size: int = 100,
    record_video: bool = False,
    action_repeat: int = 4,
    stack_size: int = 3,
    camera_id: int = 0,
    video_interval: int = 2000,
):
    def make_env(env_name: str, sub_seed: int):
        def trunk():
            env_make_fn = partial(
                gym.make,
                render_mode="rgb_array",
                render_kwargs=dict(
                    height=img_size, width=img_size, camera_id=camera_id
                ),
            )
            try:
                env = env_make_fn(env_name)
            except gym.error.NamespaceNotFound:
                register_gymnasium_envs()
                env = env_make_fn(env_name)
            env.action_space.seed(sub_seed)
            env.observation_space.seed(sub_seed)
            env = ClipAction(env)
            env = RenderObservation(env)
            env = MaxAndSkipObservation(env, skip=action_repeat)
            env = FrameStackObservation(env, stack_size=stack_size)
            env = TransformObservation(
                env,
                lambda obs: rearrange(obs, "s h w c -> h w (s c)"),
                gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(img_size, img_size, 3 * stack_size),
                    dtype=np.uint8,
                ),
            )
            if sub_seed == seed and record_video:
                env = RecordVideo(
                    env,
                    video_folder=(Path(__file__).parent / "videos").as_posix(),
                    step_trigger=lambda t: t % video_interval == 0,
                    disable_logger=True,
                    name_prefix=f"{env_name}_{seed}",
                    fps=60,
                )
            return env

        return trunk

    env = gym.vector.AsyncVectorEnv(
        [make_env(env_name, seed + i) for i in range(num_env)], context="spawn"
    )

    env = VectorRecordEpisodeStatistics(env)
    return env
