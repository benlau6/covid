import numpy as np
from covid.dist import conv1d


def test_convolve():
    a = [1, 4, 7]
    b = [0, 0, 0.2, 0.2, 0.2, 0.2, 0.2]
    result = np.convolve(a, b, mode="full").round(1)
    expected_result = np.array([0.0, 0.0, 0.2, 1.0, 2.4, 2.4, 2.4, 2.2, 1.4]).round(1)
    assert len(result) == len(a) + len(b) - 1
    assert np.array_equal(result, expected_result)


def test_conv1d():
    a = [1.0, 4.0, 7.0, 5.0, 4.0, 3.0]
    b = [0, 0, 0.2, 0.2, 0.2, 0.2]
    result = conv1d(a, b, len_observed=len(a), delay=5)

    result = result.eval().round(1)
    expected_result = np.array([3.4, 4.0, 3.8, 2.4, 1.4, 0.6]).round(1)
    assert result.ndim == 1
    assert np.array_equal(result, expected_result)
