import time
from collections.abc import Sequence
from typing import Any, Dict, Tuple

import chex
import distrax
import dm_pix
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
import orbax.checkpoint as ocp
from chex import Array, Numeric, PRNGKey
from einops import rearrange
from flashbax.buffers.flat_buffer import TrajectoryBufferState, add_dim_to_args
from flashbax.buffers.prioritised_trajectory_buffer import TrajectoryBufferSample
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from tqdm import trange

from config import DrQConfig
from logger.logger import Logger


@chex.dataclass
class TimeStep:
    observation: Array
    action: Array
    reward: Array
    done: Array
    next_observation: Array


class CriticTrainState(TrainState):
    target_params: FrozenDict

    def soft_update(self, critic_tau):
        new_target_params = optax.incremental_update(
            self.params,
            self.target_params,
            critic_tau,
        )
        return self.replace(target_params=new_target_params)


class Encoder(nn.Module):
    features_sizes: Sequence[int] = (32, 64, 128, 256)
    kernel_sizes: Sequence[int] = (3, 3, 3, 3)
    strides: Sequence[int] = (2, 2, 2, 2)
    padding: str = "VALID"
    latent_dim: int = 50

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        assert len(self.features_sizes) == len(self.strides)

        x = obs.astype(jnp.float32) / 255.0

        for features, kernel_size, stride in zip(
            self.features_sizes, self.kernel_sizes, self.strides
        ):
            x = nn.Conv(
                features,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                kernel_init=nn.initializers.he_uniform(),
                padding=self.padding,
            )(x)
            x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)

        net = nn.Sequential(
            [
                nn.Dense(
                    self.latent_dim,
                    kernel_init=nn.initializers.he_uniform(),
                ),
                nn.LayerNorm(),
                nn.tanh,
            ]
        )
        x = net(x)

        return x


class TanhNormal(distrax.Transformed):
    def __init__(self, loc: Numeric, scale: Numeric):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


# WARN: only for [-1, 1] action bounds,
# scaling/unscaling is left as an exercise for the reader :D
class Actor(nn.Module):
    action_dim: int
    encoder: nn.Module
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, state: Array):
        net = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=nn.initializers.he_uniform(),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=nn.initializers.he_uniform(),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
            ]
        )
        log_sigma_net = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(0.1),
        )
        mu_net = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.he_uniform(),
            bias_init=nn.initializers.constant(0.1),
        )
        latent = self.encoder(state)
        latent = jax.lax.stop_gradient(latent)
        trunk = net(latent)
        mu, log_sigma = mu_net(trunk), log_sigma_net(trunk)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        dist = TanhNormal(mu, jnp.exp(log_sigma))
        return dist


class Critic(nn.Module):
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, state, action):
        net = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=nn.initializers.he_uniform(),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=nn.initializers.he_uniform(),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.Dense(
                    1,
                    kernel_init=nn.initializers.uniform(1e-3),
                    bias_init=nn.initializers.uniform(1e-3),
                ),
            ]
        )
        state_action = jnp.hstack([state, action])
        out = net(state_action).squeeze(-1)
        return out


class DoubleCritic(nn.Module):
    encoder: nn.Module
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, observation, action):
        double_critic = nn.vmap(
            target=Critic,
            in_axes=None,  # type: ignore
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=2,
        )
        observation = self.encoder(observation)
        q_values = double_critic(self.hidden_dim)(observation, action)
        return q_values


class Alpha(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param(
            "log_alpha", lambda _: jnp.array([jnp.log(self.init_value)])
        )
        return jnp.exp(log_alpha)


class ConstantAlpha(nn.Module):
    init_value: float = 0.05

    @nn.compact
    def __call__(self):
        self.param("dummy_param", lambda _: jnp.full((), self.init_value))
        return self.init_value


def random_crop(key: PRNGKey, image: Array, padding):
    chex.assert_rank(image, {3, 4})
    height, width = image.shape[-3], image.shape[-2]  # type: ignore
    padded_image = dm_pix.pad_to_size(
        image, height + padding * 2, width + padding * 2, mode="edge"
    )
    return dm_pix.random_crop(key, padded_image, crop_sizes=image.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    # vmap version for jit
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def tie_encoder(source: Dict | FrozenDict, target: Dict | FrozenDict):
    target["params"]["encoder"] = source["params"]["encoder"]


# SAC losses
def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    alpha: TrainState,
    batch: TimeStep,
) -> Tuple[TrainState, Dict[str, Any]]:
    def actor_loss_fn(actor_params):
        observations = batch.observation
        actions_dist = actor.apply_fn(actor_params, observations)
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=key)

        q_values = critic.apply_fn(critic.params, observations, actions).mean(0)
        loss = (alpha.apply_fn(alpha.params) * actions_logp.sum(-1) - q_values).mean()

        batch_entropy = -actions_logp.sum(-1).mean()
        return loss, batch_entropy

    (loss, batch_entropy), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        actor.params
    )
    new_actor = actor.apply_gradients(grads=grads)
    info = {"batch_entropy": batch_entropy, "actor_loss": loss}
    return new_actor, info


def update_alpha(
    alpha: TrainState, entropy: float, target_entropy: float
) -> Tuple[TrainState, Dict[str, Any]]:
    def alpha_loss_fn(alpha_params):
        alpha_value = alpha.apply_fn(alpha_params)
        loss = (alpha_value * (entropy - target_entropy)).mean()
        return loss

    loss, grads = jax.value_and_grad(alpha_loss_fn)(alpha.params)
    new_alpha = alpha.apply_gradients(grads=grads)
    info = {"alpha": alpha.apply_fn(alpha.params), "alpha_loss": loss}
    return new_alpha, info


def update_critic(
    key: PRNGKey,
    actor: TrainState,
    critic: CriticTrainState,
    alpha: TrainState,
    batch: TimeStep,
    gamma: float,
    critic_tau: float,
) -> Tuple[CriticTrainState, Dict[str, Any]]:
    observations, next_observations = (
        batch.observation,
        batch.next_observation,
    )
    actions, rewards, dones = (
        batch.action,
        batch.reward,
        batch.done,
    )
    next_actions_dist = actor.apply_fn(actor.params, next_observations)
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=key)

    next_q = critic.apply_fn(
        critic.target_params, next_observations, next_actions
    ).mean(0)
    # backup entropy
    next_q = next_q - alpha.apply_fn(alpha.params) * next_actions_logp.sum(-1)
    target_q = rewards + (1 - dones) * gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = critic.apply_fn(critic_params, observations, actions)
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss

    loss, grads = jax.value_and_grad(critic_loss_fn)(critic.params)
    new_critic = critic.apply_gradients(grads=grads).soft_update(critic_tau=critic_tau)
    info = {"critic_loss": loss}
    return new_critic, info


# train
@jax.jit
def train_actions_jit(actor: TrainState, obs: Array, key: PRNGKey) -> Array:
    dist = actor.apply_fn(actor.params, obs)
    _, sample_key = jax.random.split(key)
    action = dist.sample(seed=sample_key)
    return action


# evaluation
@jax.jit
def eval_actions_jit(actor: TrainState, obs: Array) -> Array:
    dist = actor.apply_fn(actor.params, obs)
    action = dist.mean()
    return action


def make_buffer(
    start_steps: int,
    buffer_size: int,
    batch_size: int,
    add_batch_size: int = 1,
):
    buffer_module = fbx.make_trajectory_buffer(
        add_batch_size=add_batch_size,
        max_length_time_axis=buffer_size // add_batch_size,
        min_length_time_axis=start_steps // add_batch_size,
        sample_batch_size=batch_size,
        sample_sequence_length=1,
        period=1,
    )

    add_fn = buffer_module.add

    add_fn = add_dim_to_args(add_fn, axis=0, starting_arg_index=1, ending_arg_index=2)

    def sample_fn(
        state: TrajectoryBufferState, rng_key: PRNGKey
    ) -> TrajectoryBufferSample:
        """Samples a batch of transitions from the buffer."""
        sampled_batch = buffer_module.sample(state, rng_key).experience
        squeezed_batch = jax.tree_util.tree_map(lambda x: x[:, 0], sampled_batch)
        return TrajectoryBufferSample(experience=squeezed_batch)

    # inplace update
    return buffer_module.replace(  # type: ignore
        init=jax.jit(buffer_module.init),
        add=jax.jit(add_fn, donate_argnums=0),
        sample=jax.jit(sample_fn),
        can_sample=jax.jit(buffer_module.can_sample),
    )


def random_rollot(
    env,
    num_steps: int,
    observation: npt.NDArray,
) -> Tuple[TimeStep, npt.NDArray]:
    # env is auto-reset, so take it carefully!
    observations, actions, rewards, dones, next_observations = [], [], [], [], []
    for _ in range(num_steps):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(np.asarray(terminated, dtype=np.float32))
        real_next_observation = np.copy(next_observation)
        # handle `done` case
        for idx, done in enumerate(terminated | truncated):
            if done:
                real_next_observation[idx] = info["final_observation"][idx]
        next_observations.append(real_next_observation)
        observation = next_observation

    return (
        TimeStep(
            observation=rearrange(
                jnp.asarray(observations), "l n h w c -> (l n) h w c"
            ),
            next_observation=rearrange(
                jnp.asarray(next_observations), "l n h w c -> (l n) h w c"
            ),
            action=rearrange(jnp.asarray(actions), "l n d -> (l n) d"),
            reward=rearrange(jnp.asarray(rewards), "l n -> (l n)"),
            done=rearrange(jnp.asarray(dones), "l n -> (l n)"),
        ),
        observation,
    )


def rollout(
    env,
    actor: TrainState,
    num_steps: int,
    observation: npt.NDArray,
    key: PRNGKey,
) -> Tuple[TimeStep, npt.NDArray]:
    # env is auto-reseted, so take carefully!
    observations, actions, rewards, dones, next_observations = [], [], [], [], []
    for _ in range(num_steps):
        action = train_actions_jit(actor, observation, key)
        action = jax.device_get(action)
        next_observation, reward, terminated, truncated, info = env.step(
            action.squeeze()
        )
        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(np.asarray(terminated, dtype=np.float32))
        real_next_observation = np.copy(next_observation)
        # handle `done` case
        for idx, done in enumerate(terminated | truncated):
            if done:
                real_next_observation[idx] = info["final_observation"][idx]
        next_observations.append(real_next_observation)
        observation = next_observation

    return (
        TimeStep(
            observation=rearrange(
                jnp.asarray(observations), "l n h w c -> (l n) h w c"
            ),
            next_observation=rearrange(
                jnp.asarray(next_observations), "l n h w c -> (l n) h w c"
            ),
            action=rearrange(jnp.asarray(actions), "l n d -> (l n) d"),
            reward=rearrange(jnp.asarray(rewards), "l n -> (l n)"),
            done=rearrange(jnp.asarray(dones), "l n -> (l n)"),
        ),
        observation,
    )


def evaluate(env, actor: TrainState, num_episodes: int) -> npt.NDArray:
    observation, _ = env.reset()

    episode_cnt = 0
    while episode_cnt < num_episodes:
        action = eval_actions_jit(actor, observation)
        observation, _, terminated, truncated, _ = env.step(
            jax.device_get(action).squeeze()
        )
        episode_cnt += np.count_nonzero(terminated | truncated)
    # we use RecordEpisodeStatistics wrapper
    returns = [env.return_queue.pop().item() for _ in range(num_episodes)]
    return np.array(returns)


@hydra.main(version_base=None, config_path=".", config_name="drq_config")
def main(cfg: DrQConfig):
    # jax.config.update("jax_disable_jit", True)
    print(OmegaConf.to_yaml(cfg))

    train_env = make_vector_env(
        cfg.env_name,
        seed=cfg.eval_seed,
        num_env=cfg.num_env,
        img_size=cfg.img_size,
        action_repeat=cfg.action_repeat,
        camera_id=cfg.camera_id,
    )
    eval_env = make_vector_env(
        cfg.env_name,
        seed=cfg.eval_seed,
        num_env=cfg.eval_episodes,
        img_size=cfg.img_size,
        action_repeat=cfg.action_repeat,
        camera_id=cfg.camera_id,
        record_video=True,
        video_interval=(cfg.steps_per_epoch * cfg.eval_every) // cfg.action_repeat,
    )
    single_observation_space = train_env.single_observation_space
    single_action_space = train_env.single_action_space
    assert isinstance(single_action_space.shape, Tuple)

    logger = Logger(exp_dir=cfg.run_dir)

    target_entropy = -float(np.prod(single_action_space.shape))

    key = jax.random.PRNGKey(seed=cfg.train_seed)
    key, actor_key, critic_key, alpha_key = jax.random.split(key, 4)

    init_observation = jnp.asarray(single_observation_space.sample())
    init_action = jnp.asarray(single_action_space.sample())
    init_reward = jnp.float32(0.0)
    init_done = jnp.float32(0.0)
    init_timestep = TimeStep(
        observation=init_observation,
        action=init_action,
        reward=init_reward,
        done=init_done,
        next_observation=init_observation,
    )
    buffer_module = make_buffer(
        start_steps=cfg.start_steps,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
    )
    buffer = buffer_module.init(init_timestep)

    init_observation = jnp.expand_dims(init_observation, axis=0)
    init_action = jnp.expand_dims(init_action, axis=0)

    encoder_module = Encoder(
        latent_dim=cfg.latent_dim,
        features_sizes=cfg.features_sizes,
        kernel_sizes=cfg.kernel_sizes,
        strides=cfg.strides,
    )

    critic_module = DoubleCritic(
        encoder=encoder_module,
        hidden_dim=cfg.hidden_dim,
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_observation, init_action),
        target_params=critic_module.init(critic_key, init_observation, init_action),
        tx=optax.adam(learning_rate=cfg.critic_learning_rate),
    )

    actor_module = Actor(
        encoder=encoder_module,
        action_dim=int(np.prod(single_action_space.shape)),
        hidden_dim=cfg.hidden_dim,
    )
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_observation),
        tx=optax.adam(learning_rate=cfg.actor_learning_rate),
    )

    alpha_module = (
        Alpha(init_value=cfg.init_alpha)
        if cfg.auto_alpha
        else ConstantAlpha(init_value=cfg.init_alpha)
    )
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=cfg.alpha_learning_rate),
    )

    options = ocp.CheckpointManagerOptions(
        step_prefix="ckpt",
        max_to_keep=1,
    )
    mngr = ocp.CheckpointManager(
        f"{cfg.run_dir}/ckpts", ocp.PyTreeCheckpointer(), options=options
    )

    def update_networks(
        key: PRNGKey,
        actor: TrainState,
        critic: CriticTrainState,
        alpha: TrainState,
        batch: TimeStep,
    ):
        actor_key, critic_key = jax.random.split(key, 2)
        new_critic, critic_info = update_critic(
            critic_key, actor, critic, alpha, batch, cfg.gamma, cfg.critic_tau
        )
        new_actor, actor_info = update_actor(actor_key, actor, critic, alpha, batch)
        new_alpha, alpha_info = update_alpha(
            alpha, actor_info["batch_entropy"], target_entropy
        )

        return (
            new_actor,
            new_critic,
            new_alpha,
            {**actor_info, **critic_info, **alpha_info},
        )

    @jax.jit
    def update_step(_, carry):
        key, update_key, batch_key = jax.random.split(carry["key"], 3)
        batch = buffer_module.sample(carry["buffer"], batch_key).experience
        batch.observation = batched_random_crop(key, batch.observation)
        batch.next_observation = batched_random_crop(key, batch.next_observation)
        tie_encoder(source=carry["critic"].params, target=carry["actor"].params)

        actor, critic, alpha, update_info = update_networks(
            key=update_key,
            actor=carry["actor"],
            critic=carry["critic"],
            alpha=carry["alpha"],
            batch=batch,
        )
        update_info = jax.tree_map(
            lambda c, u: c + u, carry["update_info"], update_info
        )
        carry.update(
            key=key,
            actor=actor,
            critic=critic,
            alpha=alpha,
            update_info=update_info,
        )

        return carry

    update_carry = {
        "key": key,
        "actor": actor,
        "critic": critic,
        "alpha": alpha,
        "buffer": buffer,
    }

    total_timesteps = 0
    observation, _ = train_env.reset()
    timestep, observation = random_rollot(
        env=train_env, num_steps=cfg.start_steps // cfg.num_env, observation=observation
    )

    buffer = buffer_module.add(buffer, timestep)

    start_time = time.time()
    for epoch in trange(cfg.num_epochs + 1, desc="Epoch"):
        # metrics for accumulation during epoch and logging to wandb,
        # we need to reset them every epoch
        update_carry["update_info"] = {
            "critic_loss": jnp.array([0.0]),
            "actor_loss": jnp.array([0.0]),
            "alpha_loss": jnp.array([0.0]),
            "alpha": jnp.array([0.0]),
            "batch_entropy": jnp.array([0.0]),
        }
        update_carry.update(buffer=buffer)
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=cfg.steps_per_epoch // cfg.action_repeat,
            body_fun=update_step,
            init_val=update_carry,
        )
        # log mean over epoch for each metric
        update_info = jax.tree_map(
            lambda v: v[0].item() / cfg.steps_per_epoch,
            update_carry["update_info"],
        )

        if epoch % cfg.eval_every == 0 or epoch == cfg.num_epochs:
            eval_returns = evaluate(
                eval_env,
                update_carry["actor"],
                cfg.eval_episodes,
            )
            logger.log(
                {
                    "timesteps": total_timesteps,
                    "average_score": np.mean(eval_returns),
                    **update_info,
                    "duration": (time.time() - start_time),
                }
            )
            mngr.save(
                step=total_timesteps,
                items={
                    "actor": actor,
                    "critic": critic,
                    "alpha": alpha,
                },
            )

        timestep, observation = rollout(
            env=train_env,
            actor=update_carry["actor"],
            num_steps=cfg.steps_per_epoch // (cfg.num_env * cfg.action_repeat),
            observation=observation,
            key=key,
        )
        buffer = buffer_module.add(buffer, timestep)
        total_timesteps += int(np.prod(timestep.done.shape)) * cfg.action_repeat

    train_env.close()
    eval_env.close()
    logger.close()


if __name__ == "__main__":
    import os

    from absl import logging as absl_logging

    # disable annoying warnings in jax
    absl_logging.set_verbosity(absl_logging.FATAL)

    # disable pre-allocate gpu memories in jax
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # headless and hardware-accelerated mujoco render
    os.environ["MUJOCO_GL"] = "egl"

    # correct gpu devices for mujoco egl render in slurm cluster
    available_gpus = os.getenv("SLURM_STEP_GPUS")
    if available_gpus:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = available_gpus[0]

    from env import make_vector_env

    main()
