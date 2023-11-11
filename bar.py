from jax import numpy as jnp
import jax

def fluff(a: jnp.ndarray) -> jnp.ndarray:
    # jax.debug.print(f'assfuck {a}')
    # print('meow')
    jax.debug.breakpoint()
    return a + a

x = jnp.arange(3.)
print(fluff(x))
jx = jax.jit(fluff)
print(jx(x))
print(jx(-x))
print(jx(3 * x))
print(jx(10 * x))
