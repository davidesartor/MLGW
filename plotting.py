import matplotlib.pyplot as plt
import numpy as np


def plot_sources_signal(sources, t: np.ndarray, label=None, color=None, single=False):
    signals = np.array([source.signal(t) for source in sources])
    if not single:
        signals = np.sum(signals, axis=0)[None, :]
    for signal in signals[:-1]:
        plt.step(t, signal, c=color, label=label)
    plt.step(t, signals[-1], c=color, label=label)
    plt.title("signal")
    plt.xlabel("t")
    if label is not None:
        plt.legend()


def plot_sources_parameters(sources, label=None):
    thetas = np.array([source.theta for source in sources])
    lambdas = np.array([source.lambd for source in sources])
    periods, phases = thetas[:, 0], thetas[:, 1]
    plt.scatter(phases, periods, label=label)
    plt.title("parameters")
    plt.xlabel("shift")
    plt.ylabel("period")
    if label is not None:
        plt.legend()
    plt.xlim(-6, 6)
    plt.ylim(0.0, 2.5)


def plot_multi(sources, t: np.ndarray, new_fig=True, label=None, color=None):
    if new_fig:
        plt.figure(figsize=(9, 2))
    plt.subplot(1, 3, 2)
    plot_sources_signal(sources, t, single=False, label=label, color=color)
    plt.title("total signal")
    plt.subplot(1, 3, 3)
    plot_sources_signal(sources, t, single=True, label=label, color=color)
    plt.title("single signals")
    plt.subplot(1, 3, 1)
    plot_sources_parameters(sources, label=label)


def plot_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.semilogy(loss_log)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()