import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from covid.dist import get_convolution_ready_gt, get_generation_time_interval, conv1d


def exp_model(confirmed, days):
    with pm.Model() as exp_model:
        days_data = pm.Data("days", days)
        confirmed_data = pm.Data("confirmed", confirmed)

        a = pm.TruncatedNormal("a", mu=100, sigma=25, lower=100)
        b = pm.TruncatedNormal("b", mu=0.3, sigma=0.1, lower=0)

        growth = a * (1 + b) ** days_data
        variance = pm.Gamma("alpha", mu=6, sigma=1)

        pm.NegativeBinomial("obs", mu=growth, alpha=variance, observed=confirmed_data)
    return exp_model


def logistic_model(confirmed, days):
    with pm.Model() as logistic_model:
        days_data = pm.Data("days", days)
        confirmed_data = pm.Data("confirmed", confirmed)

        a = pm.TruncatedNormal("a", mu=100, sigma=25, lower=100)
        b = pm.TruncatedNormal("b", mu=0.3, sigma=0.1, lower=0)

        # how many ppl until the leveling off effect
        carrying_capacity = pm.Uniform("carrying_capacity", lower=1000, upper=7_000_000)
        a = carrying_capacity / a - 1

        growth = carrying_capacity / (1 + a * np.exp(-b * days_data))
        variance = pm.Gamma("alpha", mu=6, sigma=1)

        pm.NegativeBinomial("obs", mu=growth, alpha=variance, observed=confirmed_data)
    return logistic_model


def get_r_naught_model(
    new_cases, coords, add_delay: bool = False, add_onset: bool = False
) -> pm.Model:
    # the time it takes for the primary person to infect others follows a distribution.
    # They might infect one person the next day, two the day after etc.
    # This delay distribution is officially known as the "generation time"
    # we take max infection length as 7
    # ref: https://www.bbc.com/zhongwen/trad/science-60068745
    # NOTE: it affects the resulting time series inference a lot
    convolution_ready_gt = get_convolution_ready_gt(
        len(new_cases),
        max_infection_length=7,
    )

    # coords states the dimension of the data
    # coords acts like hue, or any dimension that we want to compare with
    with pm.Model(coords=coords) as r_naught_model:
        # pm.Data indicates mutability, i.e. we can change the value of the data later for forecasting
        # to change the value of the data, we can use pm.set_data
        # ref: https://www.pymc.io/projects/examples/en/latest/fundamentals/data_container.html#using-data-containers-to-mutate-data
        new_case_data = pm.Data("confirmed", new_cases)
        length_observed = new_case_data.shape[0]

        # time-varying reproduction number
        init_dist = pm.Normal.dist(mu=0, sigma=1)
        log_rt = pm.GaussianRandomWalk(
            "log_rt",
            # how quickly the reproduction number can change in one day
            # kind of a step size
            # setting it as an unusally high value to make it more flexible to react to outbreak / lockdown intervention
            # Alternative 1: using a student t distribution to make it more robust to outliers might also be a good idea
            # but currently it is not used because it samples too slow
            # Alternative 2: using a change point model might also be a good idea to detect intervention
            # and fit the model better by breaking the time series into different segments
            sigma=0.3,
            shape=length_observed,
            init_dist=init_dist,
            dims="dates",
        )
        rt = pm.Deterministic("rt", pm.math.exp(log_rt), dims="dates")
        # sometimes log normal might go wildly high after taking exp,
        # so we clip rt to a reasonable range
        rt = pt.clip(rt, 0, 10)

        initial_infected = new_cases[0]
        # seed population
        seed = pm.TruncatedNormal(
            "seed", mu=initial_infected, sigma=initial_infected, lower=0
        )
        # NOTE: it assumes the days intervals are always 1, i.e. 1, 2, .., n
        y0 = pt.zeros(length_observed)
        y0 = pt.set_subtensor(y0[0], seed)
        # y0[0] is already set to seed, so we start from 1
        ts = pt.arange(1, length_observed)

        if add_delay:
            fn = lambda gt, t, y, rt: pt.set_subtensor(y[t], pt.sum(gt * rt * y))
            sequences = [convolution_ready_gt, ts]
        else:
            fn = lambda t, y, rt: pt.set_subtensor(y[t], pt.sum(rt * y))
            sequences = [ts]

        outputs, _ = pytensor.scan(
            fn=fn,
            sequences=sequences,
            outputs_info=y0,
            non_sequences=rt,
            n_steps=length_observed - 1,
        )

        infections = pm.Deterministic("infections", outputs[-1], dims="dates")
        # clip it to avoid cap it negative binomial failure
        infections = pt.clip(infections, 0, 300000)

        if add_onset:
            # how long it takes from getting infected to test positive
            # generation time distribution is similar to the onset delay distribution
            # which is the delay from infection and confirmed positive test
            # so it is used as a proxy
            # NOTE: the delay affects the resulting time series inference a lot
            # it shifts the fitted model left or right
            p_delay = get_generation_time_interval(max_length=7)
            test_adjusted_positive = pm.Deterministic(
                "test_adjusted_positive",
                conv1d(infections, p_delay, delay=5),
                dims="dates",
            )
            mu = test_adjusted_positive + 0.1
        else:
            mu = infections

        pm.NegativeBinomial(
            "obs",
            mu=mu,
            alpha=pm.Gamma("alpha", mu=6, sigma=1),
            observed=new_case_data,
            dims="dates",
        )
    return r_naught_model
