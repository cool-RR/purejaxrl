from __future__ import annotations

from typing import Sequence, NamedTuple, Any, TypeVar
import dataclasses
import functools
import sys

import jax
import jax.numpy as jnp
import flax.linen
import numpy as np
import optax
import flax
from flax.linen import initializers as flax_initializers
import flax.training.train_state
import distrax
import gymnax
import purejaxrl.wrappers

RealNumber = int | float
EnvState = TypeVar('EnvState')
sys.breakpointhook = jax.debug.breakpoint


# @dataclasses.dataclass(kw_only=True, repr=False)
@flax.struct.dataclass
class FooConfig:
    lr: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    total_timesteps: float = 3e5
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = 'tanh'
    env_name: str = 'CartPole-v1'
    anneal_lr: bool = True
    debug: bool = True
    rng_key: int = 30

    @property
    @functools.cache
    def rng(self) -> jax.random.PRNGKey:
        return jax.random.PRNGKey(self.rng_key)

    @property
    @functools.cache
    def num_updates(self) -> int:
        return self.total_timesteps // self.num_steps // self.num_envs

    @property
    @functools.cache
    def minibatch_size(self) -> int:
        return self.num_envs * self.num_steps // self.num_minibatches


    def __hash__(self) -> int:
        return hash(
            (type(self),
             *(getattr(self, field) for field in self.__dataclass_fields__))
        )


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@flax.struct.dataclass
class RunnerState:
    train_state: flax.training.train_state.TrainState
    env_state: EnvState
    last_obs: jnp.ndarray
    rng: jax.random.PRNGKey


@flax.struct.dataclass
class UpdateState:
    train_state: flax.training.train_state.TrainState
    traj_batch: Transition # It's actually a stacked transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jax.random.PRNGKey



class ActorCritic(flax.linen.Module):
    action_dim: Sequence[int]
    activation: str = 'tanh'

    @flax.linen.compact
    def __call__(self, x):
        if self.activation == 'relu':
            activation = flax.linen.relu
        else:
            activation = flax.linen.tanh
        actor_mean = flax.linen.Dense(
            64,
            kernel_init=flax_initializers.orthogonal(np.sqrt(2)),
            bias_init=flax_initializers.constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = flax.linen.Dense(
            64,
            kernel_init=flax_initializers.orthogonal(np.sqrt(2)),
            bias_init=flax_initializers.constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = flax.linen.Dense(
            self.action_dim,
            kernel_init=flax_initializers.orthogonal(0.01),
            bias_init=flax_initializers.constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = flax.linen.Dense(
            64,
            kernel_init=flax_initializers.orthogonal(np.sqrt(2)),
            bias_init=flax_initializers.constant(0.0),
        )(x)
        critic = activation(critic)
        critic = flax.linen.Dense(
            64,
            kernel_init=flax_initializers.orthogonal(np.sqrt(2)),
            bias_init=flax_initializers.constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = flax.linen.Dense(
            1,
            kernel_init=flax_initializers.orthogonal(1.0),
            bias_init=flax_initializers.constant(0.0)
        )(critic)


        return pi, jnp.squeeze(critic, axis=-1)


class LinearSchedule:
    def __init__(self, foo_config: FooConfig) -> None:
        self.foo_config = foo_config

    def __call__(self, count: int) -> RealNumber:
        frac = (
            1.0
            - (count // (self.foo_config.num_minibatches * self.foo_config.update_epochs))
            / self.foo_config.num_updates
        )
        return self.foo_config.lr * frac


def make_train(foo_config: FooConfig):
    env, env_params = gymnax.make(foo_config.env_name)
    env = purejaxrl.wrappers.FlattenObservationWrapper(env)
    env = purejaxrl.wrappers.LogWrapper(env)

    def train(rng: jax.random.PRNGKey) -> tuple[RunnerState, dict]:
        '''
        return runner_state, metric
        '''
        # Init network
        rng = foo_config.rng
        network = ActorCritic(
            env.action_space(env_params).n, activation=foo_config.activation
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if foo_config.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(foo_config.max_grad_norm),
                optax.adam(learning_rate=LinearSchedule(foo_config), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(foo_config.max_grad_norm),
                optax.adam(foo_config.lr, eps=1e-5),
            )
        train_state = flax.training.train_state.TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, foo_config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Train loop
        def _update_step(runner_state: RunnerState, unused: None):
            # Collect trajectories
            def _env_step(runner_state: runner_state, unused: None) -> tuple[RunnerState,
                                                                             Transition]:
                # Select action
                rng, _rng = jax.random.split(runner_state.rng)
                pi, value = network.apply(runner_state.train_state.params, runner_state.last_obs)
                action = pi.sample(seed=_rng) # E.g. array([0, 1, 0, 1])
                log_prob = pi.log_prob(action) # E.g. array([-0.1, -0.2, -0.4, -0.2], dtype=float32)
                # breakpoint()

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, foo_config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, runner_state.env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, runner_state.last_obs, info
                )
                return (
                    RunnerState(train_state=runner_state.train_state, env_state=env_state,
                                last_obs=obsv, rng=rng),
                    transition,
                )

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, foo_config.num_steps
            )


            # Calculate advantage
            _, last_val = network.apply(runner_state.train_state.params, runner_state.last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + foo_config.gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + foo_config.gamma * foo_config.gae_lambda * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # Update network
            def _update_epoch(update_state: UpdateState, unused: None) -> tuple[UpdateState, None]:
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_function(params, traj_batch, gae, targets):
                        # Rerun network
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-foo_config.clip_eps, foo_config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - foo_config.clip_eps,
                                1.0 + foo_config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + foo_config.vf_coef * value_loss
                            - foo_config.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # End of _loss_function, continuing _update_minibatch


                    grad_fn = jax.value_and_grad(_loss_function, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                # End of _update_minibatch, continuing _update_epoch

                rng, _rng = jax.random.split(update_state.rng)
                batch_size = foo_config.minibatch_size * foo_config.num_minibatches
                assert (
                    batch_size == foo_config.num_steps * foo_config.num_envs
                ), 'batch size must be equal to number of steps * number of envs'
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (update_state.traj_batch, update_state.advantages, update_state.targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [foo_config.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minibatch, update_state.train_state, minibatches
                )
                return (
                    UpdateState(train_state=train_state, traj_batch=update_state.traj_batch,
                                advantages=update_state.advantages, targets=update_state.targets,
                                rng=rng),
                    total_loss
                )

            # End of _update_epoch, continuing _update_step


            update_state = UpdateState(train_state=runner_state.train_state, traj_batch=traj_batch,
                                       advantages=advantages, targets=targets, rng=runner_state.rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, foo_config.update_epochs
            )
            metric = traj_batch.info
            if foo_config.debug:
                def callback(info):
                    return_values = info['returned_episode_returns'][info['returned_episode']]
                    timesteps = info['timestep'][info['returned_episode']] * foo_config.num_envs
                    for t in range(len(timesteps)):
                        print(f'global step={timesteps[t]}, episodic return={return_values[t]}')
                jax.debug.callback(callback, metric)

            return (
                RunnerState(
                    train_state=update_state.train_state, env_state=runner_state.env_state,
                    last_obs=runner_state.last_obs, rng=update_state.rng),
                metric,
            )

        # End of _update_step, continuing train

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv,
                                   rng=_rng)
        # breakpoint()
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, foo_config.num_updates
        )
        return {'runner_state': runner_state, 'metrics': metric}

    # End of train, continuing make_train

    return train



if __name__ == '__main__':
    foo_config = FooConfig()
    train_jit = jax.jit(make_train(foo_config))
    out = train_jit(foo_config.rng)
