""" 
"""
#from pdb import set_trace

import jax
import jax.numpy as jnp
from jax import jacfwd  #, combinations

jax.config.update("jax_enable_x64", True)

from functools import partial

from jax import jit, lax, random, tree_util, vmap
from jax.experimental import host_callback

def is_number(in_value):
    try:
        float(in_value)
        return True
    except ValueError:
        return False

@jit
def choose_sk(n: jnp.int16, k: jnp.int16) -> jnp.int32:
    def choose_hlp_sk(n: jnp.int16, k: jnp.int16) -> jnp.int32:
        delta, imax = lax.cond(
            k < n - k,
            lambda: (jnp.int16(n - k), jnp.int16(k)),
            lambda: (jnp.int16(k), jnp.int16(n - k)),
        )

        return lax.fori_loop(
            jnp.int32(2),
            jnp.int32(imax + 1),
            lambda i, carry: jnp.int32((carry * (delta + i)) // i),
            jnp.int32(delta + 1),
        )

    return lax.cond(
        n < k,
        lambda: jnp.int32(0),
        lambda: lax.cond(
            n == k, lambda: jnp.int32(1), lambda: jnp.int32(choose_hlp_sk(n, k))
        ),
    )


@jit
def largestV_sk(a: jnp.int16, b: jnp.int16, x_sk: jnp.int32) -> jnp.array:
    def cond(val):
        return choose_sk(val, b) > x_sk

    def body(val):
        return jnp.int16(val - 1)

    return lax.while_loop(cond_fun=cond, body_fun=body, init_val=jnp.int16(a - 1))


@partial(jit, static_argnums=(0,1,), )
def cuda_calculateMth(n: jnp.int16, k: jnp.int16, m: jnp.array) -> jnp.array:
    # print(f"cuda_calculateMth({n}, {k}, {m.shape}:{m.dtype})")
    def choose_hlp(n_sk: jnp.int16, k_sk: jnp.int16) -> jnp.int32:
        delta, imax = lax.cond(
            k_sk < n_sk - k_sk,
            lambda: (jnp.int16(n_sk - k_sk), jnp.int16(k_sk)),
            lambda: (jnp.int16(k_sk), jnp.int16(n_sk - k_sk)),
        )

        return lax.fori_loop(
            jnp.int32(2),
            jnp.int32(imax + 1),
            lambda i, carry: jnp.int32((carry * (delta + i)) // i),
            jnp.int32(delta + 1),
        )

    def calcR(i, carry):
        d_result, x, a, b = carry
        lv = jax.vmap(largestV_sk, (0, None, 0))(a, b, x)
        # jax.debug.print("i={i}  🤯 {lv} 🤯", i=i, lv=lv)
        d_result = d_result.at[:, i].set(lv)
        x -= jnp.select(
            condlist=[lv > b, lv == b],
            choicelist=[jnp.vectorize(choose_hlp)(lv, b), jnp.int32(1)],
            default=jnp.int32(0),
        )
        return d_result, x, lv, b - 1

    d_result, _, _, _ = lax.fori_loop(
        0,
        k,
        calcR,
        (
            jnp.zeros((m.shape[0], k), dtype=jnp.int16),
            m,
            jnp.ones(m.shape[0], dtype=jnp.int16) * n,
            k,
        ),
    )
    return d_result  # (n-1) -
