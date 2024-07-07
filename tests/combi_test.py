# pytest -rP tests
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import math

import jax
import jax.numpy as jnp
from jax import jacfwd

from combinadics.combinadics import cuda_calculateMth, is_number

jax.config.update("jax_enable_x64", True)

import itertools
from time import time

import numpy as np


def mytest_combi(n, k):
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

    print(f"\njax elapsed time: {endTime-startTime}")

    startTime = time()
    expected = jnp.array(list(itertools.combinations(range(n), k)), dtype=jnp.int16)
    same = jnp.array_equal(expected, result)
    endTime = time()

    print(f"\nnumpy elapsed time: {endTime-startTime}")

    assert same


def test_combi_5_3():
    mytest_combi(5, 3)


def test_combi_7_2():
    mytest_combi(7, 2)


def test_combi_72_1():
    mytest_combi(72, 1)


def test_combi_72_2():
    mytest_combi(72, 2)


# def test_combi_72_3():
#     mytest_combi(72, 3)

# def test_combi_72_4():
#     mytest_combi(72, 4)


def test_combi_576_1():
    mytest_combi(576, 1)


# def test_combi_576_2():
#     mytest_combi(576, 2)

# def test_combi_576_3():
#     mytest_combi(576, 3)

# def test_combi_576_4():
# test_combi(576,4) # must use int64 for mth numbers


def test_demo():
    # setup
    n = 4
    k = 3
    totalcount = math.comb(n, k)

    # numpy
    print(f'Calculate combinations "{n} choose {k}" in numpy:')
    for comb in itertools.combinations(np.arange(start=0, stop=n, dtype=jnp.int32), k):
        print(comb)

    # combinadics
    print("Calculate via combinadics:")
    actual = (
        n
        - 1
        - cuda_calculateMth(
            n,
            k,
            totalcount - 1 - jnp.arange(start=0, stop=n, dtype=jnp.int32),
        )
    )
    for comb in actual:
        print(comb)
