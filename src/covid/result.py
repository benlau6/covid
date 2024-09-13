import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def save_trace(trace, filename):
    trace.to_netcdf(filename)


def load_trace(filename):
    return az.from_netcdf(filename)


def plot_predictive_check(samples, obs, ylim=(-100, 1000), xlim=(0, 20)):
    _, ax = plt.subplots(figsize=(12, 8))
    ax.plot(samples, "C0", alpha=0.1)
    ax.plot(obs, "r", label="data")
    ax.set(
        ylim=ylim,
        xlim=xlim,
        title="Prior predictive checks",
        xlabel="days",
        ylabel="confirmed",
    )


def plot_credible_interval(samples, xs, alpha: float = 0.05, color: str = "C0"):
    p = 100 * alpha / 2
    upper = np.percentile(samples, p, axis=1)
    lower = np.percentile(samples, 100 - p, axis=1)
    plt.fill_between(
        xs,
        upper,
        lower,
        color=color,
        alpha=0.8,
        label=f"{100-100*alpha}% CI",
    )


def plot_ts_trace():
    pass
    # ax.plot(trace.posterior.rt.mean(dim=["chain"]).T, "C0", alpha=0.1)
    # ax.plot(trace.posterior.rt.mean(dim=["chain", "draw"]).T, "darkred")
    # az.plot_hdi(
    #     days,
    #     trace.posterior.rt,
    #     ax=ax,
    #     fill_kwargs={"alpha": 0.2},
    # )
