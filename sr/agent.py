import jax
import jax.numpy as jnp
from jax.experimental.stax import (
    Dense,
    Conv,
    Relu,
    Sigmoid,
    parallel,
    serial,
    FanOut,
    Flatten,
)
from helx.nn.module import module
import helx.nn


@module
def SyntheticReturn(features_network):
    """Synthetic return module as described in:
    https://arxiv.org/abs/2102.12425,
    Raposo, D., Synthetic Returns for Long-Term Credit Assignment, 2021."""
    #  sigmoid gate
    g = lambda: serial(Dense(256), Relu, Dense(1), Relu, Dense(1), Sigmoid)
    #  state utility contribution
    c = lambda: serial(Dense(256), Relu, Dense(256), Relu, Dense(1))
    #  state utility baseline
    b = lambda: serial(Dense(256), Relu, Dense(256), Relu, Dense(1))
    return serial(features_network, Flatten, FanOut(3), parallel(g(), c(), b()))


@module
def Mpl():
    """MPL used for the Chain experiment in:
    https://arxiv.org/abs/2102.12425"""
    return serial(Dense(128), Relu)


@module
def CnnSmall():
    """CNN used for the Catch and Key-to-Door experiments in:
    https://arxiv.org/abs/2102.12425"""
    return serial(
        Conv(32, (2, 2), (1, 1), "VALID"),
        Relu,
        Conv(64, (2, 2), (1, 1), "VALID"),
        Relu,
        Flatten,
        Dense(256),
        Relu,
    )


@module
def CnnLarge():
    """CNN used for the Ponf and Skiing experiments in:
    https://arxiv.org/abs/2102.12425"""
    return serial(
        Conv(32, (3, 3), (2, 2), "VALID"),
        Relu,
        Conv(64, (3, 3), (2, 2), "VALID"),
        Relu,
        Conv(64, (3, 3), (2, 2), "VALID"),
        Relu,
        Flatten,
        Dense(256),
        Relu,
    )


@module
def PolicyNetwork():
    """Policy network for the experiments in:
    https://arxiv.org/abs/2102.12425"""
    return serial(
        helx.nn.rnn.LSTM(256),
        Dense(256),
        Relu,
        FanOut(2),
        parallel(Dense(1), Dense(1)),
    )
