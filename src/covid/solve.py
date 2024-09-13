import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pymc as pm
from pykalman import KalmanFilter

from covid.data import get_data_model
from covid.model import exp_model, get_r_naught_model, logistic_model
from covid.result import load_trace, plot_credible_interval, save_trace

# uses constrained_layout to adjust the plot layout for trace plot
# constrained_layout is similar to tight_layout,
# but uses a constraint solver to determine the size of axes that allows them to fit.
# ref: https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
plt.rcParams["figure.constrained_layout.use"] = True


def model_comparison(confirmed, days):
    with get_r_naught_model(confirmed, days):
        pm.set_data({"days": days, "confirmed": confirmed})
        r_naught_full = pm.sample(**fitting_params)
    with exp_model(confirmed, days):
        pm.set_data({"days": days, "confirmed": confirmed})
        exp_full = pm.sample(**fitting_params)

    with logistic_model(confirmed, days):
        pm.set_data({"days": days, "confirmed": confirmed})
        logistic_full = pm.sample(**fitting_params)

    # weight indicates how we should put weight to make an ensemble model
    # 1.0 vs 0.0 means we should put 100% weight on the first model
    result = az.compare(
        {
            "exp": exp_full,
            "logistic": logistic_full,
            "r_naught": r_naught_full,
        },
    )
    print(result)


def smooth_obs(obs, Model):
    kf = Model()
    return kf.em(obs).smooth(obs)[0].flatten().astype(int)


fitting_params = {
    "draws": 500,
    "tune": 1000,
    "chains": 8,
    "cores": 8,
    "idata_kwargs": {"log_likelihood": True},
}

if __name__ == "__main__":
    update = True
    num_days = 60
    since_val = 150
    confirmed, days, dates = get_data_model(num_days=num_days, since_val=since_val)
    confirmed = smooth_obs(confirmed, KalmanFilter)

    filename = "output/r_naught_model_trace.nc"
    coords = {"days": days, "dates": dates}
    r_naught = get_r_naught_model(
        new_cases=confirmed,
        coords=coords,
        add_delay=True,
        add_onset=True,
    )

    if update:
        with r_naught:
            prior_pred = pm.sample_prior_predictive()
            az.plot_ppc(prior_pred, group="prior")
            plt.savefig("output/prior_ppc.png")
            # pm.set_data({"days": days, "confirmed": confirmed})
            trace = pm.sample(**fitting_params)
            save_trace(trace, filename)
    else:
        trace = load_trace(filename)

    with r_naught:
        # NOTE: trace.posterior is different from trace
        # sometimes the former works, sometimes the latter
        post_pred = pm.sample_posterior_predictive(trace)
        # ref: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/posterior_predictive.html#prior-and-posterior-predictive-checks
        az.plot_ppc(post_pred)
        plt.savefig("output/ppc.png")

    az.plot_trace(trace)
    plt.savefig("output/trace.png")
    az.plot_energy(trace)
    plt.savefig("output/energy.png")

    # plot reproduction rate
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("Reds")
    samples = trace.posterior.rt.median(dim=["chain", "draw"])
    # NOTE: matplotlib somehow does not support xarray dates
    # it is needed to convert it to numpy array
    ax.plot(dates, samples, color=cmap(1.0))
    ax.set(xlabel="date", ylabel="$R_e(t)$")
    ax.axhline(1.0, c="k", lw=1, linestyle="--")
    myFmt = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(myFmt)

    # plot 95% credible interval
    samples = trace.posterior.rt.median(dim=["chain"]).T
    plot_credible_interval(samples, dates, alpha=0.05, color=cmap(0.1))

    plt.savefig("output/rt.png")
    az.plot_ts(post_pred, y="obs")
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.savefig("output/ts.png")
