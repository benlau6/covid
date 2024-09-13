import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
from scipy import stats as sts


def get_generation_time_interval(max_length: int = 7):
    # infection-delay
    # serial interval distribution
    mean_si = 4.7
    std_si = 2.9
    mu_si = np.log(mean_si**2 / np.sqrt(std_si**2 + mean_si**2))
    sigma_si = np.sqrt(np.log(std_si**2 / mean_si**2 + 1))
    generation_time_dist = sts.lognorm(scale=np.exp(mu_si), s=sigma_si)

    # discretize the generation interval up to maximum length of infection
    # it is assumed 7 days is the maximum length of infection
    g_range = np.arange(max_length)
    gt = pd.Series(generation_time_dist.cdf(g_range), index=g_range)
    gt = gt.diff().fillna(0)
    gt /= gt.sum()
    gt = gt.values
    return gt


def get_convolution_ready_gt(length, max_infection_length=10):
    # In order to include this effect in our generative model we need to do a convolution
    # We need to take all of these previously infected people into account and by which probability they infect people today.
    gt = get_generation_time_interval(max_length=max_infection_length)
    convolution_ready_gt = np.zeros((length - 1, length))
    for t in range(1, length):
        begin = np.maximum(0, t - len(gt) + 1)
        slice_update = gt[1 : t - begin + 1][::-1]
        convolution_ready_gt[t - 1, begin : begin + len(slice_update)] = slice_update
    convolution_ready_gt = pytensor.shared(convolution_ready_gt)
    return convolution_ready_gt


def conv1d(input, kernel, delay=8):
    from pytensor.tensor.conv import conv2d

    # there only is a conv2d function in pytensor
    # and it only supports 4D tensor
    # so we need to reshape the input and kernel to 4D tensor
    output = conv2d(
        pt.reshape(input, (1, 1, 1, input.shape[0])),
        pt.reshape(kernel, (1, 1, 1, kernel.shape[0])),
        border_mode="full",
    )

    output = pt.flatten(output, ndim=1)
    return output[delay : input.shape[0] + delay]
