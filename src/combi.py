""" Idea of combinations calculations is from here: 
https://jamesmccaffrey.wordpress.com/2022/06/28/generating-the-mth-lexicographical-element-of-a-combination-using-the-combinadic/
"""
from pdb import set_trace

import jax
import jax.numpy as jnp
from jax.config import config
from jax import jacfwd, combinations

config.update("jax_enable_x64", True)
from functools import partial
from time import time

from jax import jit, lax, random, tree_util, vmap
from jax.experimental import host_callback


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


@partial(
    jit,
    static_argnums=(
        0,
        1,
    ),
)
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


def test_combi(n, k):
    totalcount = 1
    factorial = 1
    for i in range(k):
        totalcount *= n - i
        factorial *= i + 1

    totalcount = totalcount // factorial

    print("_" * 80)
    print(f"n={n}  k={k}  totalcount={totalcount}")

    # largest = largestV_sk(7,4,8)
    # print(f"largestV_sk(7,4,8)={largest}    expect 5")

    # largest = largestV_sk(7,4,3)
    # print(f"largestV_sk(7,4,3)={largest}    expect 3")

    # largest = largestV(jnp.int16(7), jnp.int16(4), jnp.array([8,3], dtype=jnp.int32))
    # print(f"largestV(7,4,[8,3])={largest}    expect [5,3]")

    # nck = choose_sk(n, k)
    # print(f"nck_sk={nck}")

    # # narr = jnp.ones(5, dtype=jnp.int16) * n
    # narr = [576, 72, 10, 5, 1]
    # print(f"narr={narr} choose {k}")
    # nck = choose(jnp.array(narr), jnp.int16(k))
    # print(f"nck={nck}")

    startTime = time()
    max_chunk = 2**21
    for chunk in range((totalcount + max_chunk - 1) // max_chunk):
        start = chunk * max_chunk
        stop = start + max_chunk
        if stop > totalcount:
            stop = totalcount
        print(f"calc range {start} - {stop}")
        result = jnp.int16(n - 1) - cuda_calculateMth(
            n,
            k,
            jnp.int32(totalcount - 1)
            - jnp.arange(start=start, stop=stop, dtype=jnp.int32),
        )
    endTime = time()

    # for i in range(totalcount):
    #     print(f"\n{i}:  ", end='')
    #     for j in range(k):
    #         print(f"{result[i,j]} ", end='')

    print(f"\nelapsed time: {endTime-startTime}")

# def smoketest():
#     print(f"{jax.__version__}")

#     def sigmoid(x):
#         return 0.5 * (jnp.tanh(x / 2) + 1)

#     # Outputs probability of a label being true.
#     def predict(W, b, inputs):
#         return sigmoid(jnp.dot(inputs, W) + b)
    
#     # Build a toy dataset.
#     inputs = jnp.array([[0.52, 1.12,  0.77],
#                     [0.88, -1.08, 0.15],
#                     [0.52, 0.06, -1.30],
#                     [0.74, -2.49, 1.39]])
#     targets = jnp.array([True, True, False, True])

#     # Training loss is the negative log-likelihood of the training examples.
#     def loss(W, b):
#         preds = predict(W, b, inputs)
#         label_probs = preds * targets + (1 - preds) * (1 - targets)
#         return -jnp.sum(jnp.log(label_probs))

#     # Initialize random model coefficients
#     key = random.PRNGKey(0)
#     key, W_key, b_key = random.split(key, 3)
#     W = random.normal(W_key, (3,))
#     b = random.normal(b_key, ())

#     # Isolate the function from the weight matrix to the predictions
#     f = lambda W: predict(W, b, inputs)

#     J = jacfwd(f)(W)
#     print("jacfwd result, with shape", J.shape)
#     print(J)

if __name__ == "__main__":
    x = combinations(3)
    print(f"x={x}")

    # # test_combi(5,3)
    # # test_combi(7,2)

    # test_combi(72, 1)
    # test_combi(72, 2)
    # test_combi(72, 3)
    # test_combi(72, 4)

    # test_combi(576, 1)
    # test_combi(576, 2)
    # test_combi(576, 3)
    # # test_combi(576,4) # must use int64 for mth numbers
