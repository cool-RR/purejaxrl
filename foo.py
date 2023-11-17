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
EnvState = TypeVar('EnvState') # This is a different class for each environment, but often this will
                               # actually be `purejaxrl.wrappers.LogEnvState`.
sys.breakpointhook = jax.debug.breakpoint


@flax.struct.dataclass
class FooConfig:

    ### Defining various integers that determine how much training we'll do: #######################
    #                                                                                              #
    n_envs: int = 4
    n_steps: int = 128
    total_timesteps: float = 3e5
    n_epochs_per_aeon: int = 4
    n_minibatches: int = 4

    @property
    @functools.cache
    def n_aeons(self) -> int:
        return self.total_timesteps // self.n_steps // self.n_envs

    @property
    @functools.cache
    def batch_size(self) -> int:
        return self.minibatch_size * self.n_minibatches

    @property
    @functools.cache
    def minibatch_size(self) -> int:
        return self.n_envs * self.n_steps // self.n_minibatches
    #                                                                                              #
    ### Finished defining various integers that determine how much training we'll do. ##############


    ### Defining training parameters: ##############################################################
    #                                                                                              #
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = 'tanh'
    anneal_lr: bool = True
    #                                                                                              #
    ### Finished defining training parameters. #####################################################

    env_name: str = 'CartPole-v1'
    debug: bool = True
    rng_key: int = 30

    @property
    @functools.cache
    def rng(self) -> jax.random.PRNGKey:
        return jax.random.PRNGKey(self.rng_key)

    def __post_init__(self):
        assert self.batch_size == self.n_steps * self.n_envs


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
class TransitionCarry:
    env_state: EnvState
    last_observation: jnp.ndarray
    rng: jax.random.PRNGKey


@flax.struct.dataclass
class AeonCarry:
    train_state: flax.training.train_state.TrainState
    rng: jax.random.PRNGKey


@flax.struct.dataclass
class EpochCarry:
    train_state: flax.training.train_state.TrainState
    batched_transition: Transition # It's actually a stacked transition
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


@flax.struct.dataclass
class LinearSchedule:
    n_minibatches: int
    n_epochs_per_aeon: int
    n_aeons: int
    lr: float

    @staticmethod
    def from_foo_config(foo_config: FooConfig) -> LinearSchedule:
        return LinearSchedule(
            n_minibatches=foo_config.n_minibatches,
            n_epochs_per_aeon=foo_config.n_epochs_per_aeon,
            n_aeons=foo_config.n_aeons,
            lr=foo_config.lr,
        )


    def __call__(self, count: int) -> RealNumber:
        frac = (
            1.0
            - (count // (self.n_minibatches * self.n_epochs_per_aeon))
            / self.n_aeons
        )
        return self.lr * frac


def loss_function(actor_critic: ActorCritic, params: dict, batched_transition: Transition,
                  gae: jnp.ndarray, targets: jnp.ndarray
                  ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    # Rerun actor_critic
    pi, value = actor_critic.apply(params, batched_transition.obs)
    log_prob = pi.log_prob(batched_transition.action)

    # calculate value loss
    value_pred_clipped = batched_transition.value + (
        value - batched_transition.value
    ).clip(-foo_config.clip_eps, foo_config.clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = (
        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
    )

    # Calculate actor loss
    ratio = jnp.exp(log_prob - batched_transition.log_prob)
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


def calculate_gae(batched_transition: Transition, last_val: jnp.ndarray) -> tuple[jnp.ndarray,
                                                                                  jnp.ndarray]:
    def get_advantages(gae_and_next_value: tuple[jnp.ndarray, jnp.ndarray], transition: Transition
                       ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        next_gae, next_value = gae_and_next_value
        discounted_next_value = foo_config.gamma * next_value * (1 - transition.done)
        delta = transition.reward + discounted_next_value - transition.value
        gae = delta + (foo_config.gamma * foo_config.gae_lambda * (1 - transition.done) * next_gae)
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        get_advantages,
        (jnp.zeros_like(last_val), last_val),
        batched_transition,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + batched_transition.value


class Trainer:
    def __init__(self, foo_config: FooConfig) -> None:
        self.foo_config = foo_config
        raw_env, self.env_params = gymnax.make(foo_config.env_name)
        # self.env_params is an immutable `EnvParams` object with constants for the environment,
        # like the gravity constant.
        self.env = purejaxrl.wrappers.LogWrapper(
            purejaxrl.wrappers.FlattenObservationWrapper(raw_env)
        )
        self.actor_critic = ActorCritic(
            self.env.action_space(self.env_params).n, activation=foo_config.activation
        )

    def _vmapped_reset(self, reset_rngs, env_params):
        '''
        returns `obs, env_state`
        '''
        return jax.vmap(self.env.reset, in_axes=(0, None))(reset_rngs, env_params)


    def make_transitions(self,
                         rng: jax.random.PRNGKey,
                         train_state: flax.training.train_state.TrainState
                         ) -> tuple[jnp.ndarray, Transition]:

        def make_transition(transition_carry: TransitionCarry, unused: None
                            ) -> tuple[TransitionCarry, Transition]:
            # Select action
            rng, _rng = jax.random.split(transition_carry.rng)
            pi, value = self.actor_critic.apply(train_state.params,
                                                transition_carry.last_observation)
            action = pi.sample(seed=_rng) # E.g. array([0, 1, 0, 1])
            log_prob = pi.log_prob(action) # E.g. array([-0.1, -0.2, -0.4, -0.2], dtype=float32)

            # Step env
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, foo_config.n_envs)
            observation, env_state, reward, done, info = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(rng_step, transition_carry.env_state, action, self.env_params)
            transition = Transition(
                done, action, value, reward, log_prob, transition_carry.last_observation, info
            )
            return (
                TransitionCarry(env_state=env_state,
                                     last_observation=observation,
                                     rng=rng),
                transition,
            )

        make_transition_rng, reset_rng = jax.random.split(rng, 2)
        reset_rngs = jax.random.split(reset_rng, foo_config.n_envs)
        last_observation, env_state = self._vmapped_reset(reset_rngs, self.env_params)

        transition_carry, batched_transition = jax.lax.scan(
            make_transition,
            TransitionCarry(
                env_state=env_state,
                last_observation=last_observation,
                rng=make_transition_rng,
            ),
            None,
            foo_config.n_steps,
        )

        return last_observation, batched_transition



    def train_aeon(self, aeon_carry: AeonCarry, unused: None) -> tuple[AeonCarry, dict]:

        rng, transition_rng = jax.random.split(aeon_carry.rng)
        last_observation, batched_transition = self.make_transitions(transition_rng,
                                                                     aeon_carry.train_state)

        _, last_val = self.actor_critic.apply(aeon_carry.train_state.params,
                                              last_observation)

        advantages, targets = calculate_gae(batched_transition, last_val)

        def train_epoch(epoch_carry: EpochCarry, unused: None) -> tuple[EpochCarry, None]:
            def train_minibatch(train_state, batch_info):
                batched_transition, advantages, targets = batch_info

                grad_fn = jax.value_and_grad(
                    lambda *args: loss_function(self.actor_critic, *args),
                    has_aux=True
                )
                total_loss, grads = grad_fn(
                    train_state.params, batched_transition, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            rng, _rng = jax.random.split(epoch_carry.rng)
            permutation = jax.random.permutation(_rng, foo_config.batch_size)
            batch = (epoch_carry.batched_transition, epoch_carry.advantages,
                     epoch_carry.targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((foo_config.batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [foo_config.n_minibatches, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(
                train_minibatch, epoch_carry.train_state, minibatches
            )
            return (
                EpochCarry(train_state=train_state,
                            batched_transition=epoch_carry.batched_transition,
                            advantages=epoch_carry.advantages, targets=epoch_carry.targets,
                            rng=rng),
                total_loss
            )

        # End of train_epoch, continuing train_aeon


        epoch_carry = EpochCarry(train_state=aeon_carry.train_state,
                                 batched_transition=batched_transition,
                                 advantages=advantages, targets=targets, rng=rng)
        epoch_carry, loss_info = jax.lax.scan(
            train_epoch, epoch_carry, None, foo_config.n_epochs_per_aeon
        )
        metric = batched_transition.info
        if foo_config.debug:
            def callback(info):
                return_values = info['returned_episode_returns'][info['returned_episode']]
                timesteps = info['timestep'][info['returned_episode']] * foo_config.n_envs
                for t in range(len(timesteps)):
                    print(f'global step={timesteps[t]}, episodic return={return_values[t]}')
            jax.debug.callback(callback, metric)

        return (
            AeonCarry(train_state=epoch_carry.train_state, rng=epoch_carry.rng),
            metric,
        )



    def __call__(self):
        rng, _rng = jax.random.split(self.foo_config.rng)
        init_x = jnp.zeros(self.env.observation_space(self.env_params).shape)
        actor_critic_params = self.actor_critic.init(_rng, init_x)
        if foo_config.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(foo_config.max_grad_norm),
                optax.adam(learning_rate=LinearSchedule.from_foo_config(foo_config), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(foo_config.max_grad_norm),
                optax.adam(foo_config.lr, eps=1e-5),
            )
        train_state = flax.training.train_state.TrainState.create(
            apply_fn=self.actor_critic.apply,
            params=actor_critic_params,
            tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        aeon_carry = AeonCarry(train_state=train_state, rng=_rng)
        aeon_carry, metric = jax.lax.scan(
            self.train_aeon, aeon_carry, None, foo_config.n_aeons
        )
        return {'aeon_carry': aeon_carry, 'metrics': metric}



if __name__ == '__main__':
    foo_config = FooConfig()
    train_jit = jax.jit(Trainer(foo_config))
    out = train_jit()
