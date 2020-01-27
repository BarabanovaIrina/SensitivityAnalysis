import numpy as np
import pytest
from scipy.stats import norm
from analysis import (evaluate_elem_effect,
                      morris,
                      get_decreased_values,
                      get_increased_values,
                      compute_mu_star_conf,
                      )
from inputfactors import Container, ModelInputFactor
from sample import lhc_sample
from SALib.test_functions import Ishigami


@pytest.fixture()
def test_set_model_factors():
    np.random.seed(1)
    fac_1 = ModelInputFactor('1', -1, 1)
    fac_2 = ModelInputFactor('2', -1, 1)
    fac_3 = ModelInputFactor('3', -1, 1)
    container = list((fac_1, fac_2, fac_3))
    input_factors_container = Container(container)

    # sample
    test_input_sample = lhc_sample(100, input_factors_container, seed=1)
    test_output_sample = Ishigami.evaluate(test_input_sample)

    num_levels = 4
    delta = num_levels / (2.0 * (num_levels - 1))

    params_number = test_input_sample.shape[1]
    trajectories_num = int(test_output_sample.size / (params_number + 1))

    return input_factors_container, test_input_sample, test_output_sample, delta, trajectories_num


def test_morris(test_set_model_factors):
    # set
    np.random.seed(1)
    input_factors_container, test_input_sample, test_output_sample, delta, trajectories_num = test_set_model_factors

    # analyze
    ee = evaluate_elem_effect(test_input_sample,
                              test_output_sample,
                              trajectories_num,
                              delta)
    expected_mu_index = np.average(ee, 1)
    actual_mu_index = morris(input_factors_container, test_input_sample, test_output_sample, seed=1)['mu']

    assert expected_mu_index.all() == actual_mu_index.all()


def test_evaluate_elem_effect(test_set_model_factors):
    np.random.seed(1)

    input_factors_container, test_input_sample, test_output_sample, delta, trajectories_num = test_set_model_factors

    trajectories_size = int(test_output_sample.size / trajectories_num)

    vector = test_input_sample.reshape(trajectories_num, trajectories_size, test_input_sample.shape[1])
    cha = np.subtract(vector[:, 1:, :], vector[:, 0:-1, :])
    up = (cha > 0)
    lo = (cha < 0)

    op_vec = test_output_sample.reshape(trajectories_num, trajectories_size)

    result_up = get_increased_values(op_vec, up, lo)
    result_lo = get_decreased_values(op_vec, up, lo)

    ee = result_up - result_lo
    np.divide(ee, delta, out=ee)

    expected_result = ee
    actual_result = evaluate_elem_effect(test_input_sample, test_output_sample, trajectories_num, delta)

    assert actual_result.all() == expected_result.all()


def test_compute_mu_star_conf(test_set_model_factors):
    np.random.seed(1)

    input_factors_container, test_input_sample, test_output_sample, delta, trajectories_num = test_set_model_factors

    conf_level = 0.95
    re_sample = 1000

    ee = evaluate_elem_effect(test_input_sample, test_output_sample, trajectories_num, delta)
    resample_index = np.random.randint(len(ee), size=(re_sample, trajectories_num))
    ee_resampled = ee[resample_index]
    mu_star_resampled = np.average(np.abs(ee_resampled), axis=1)

    expected_result = norm.ppf(0.5 + conf_level / 2) * mu_star_resampled.std(ddof=1)
    absolute_result = compute_mu_star_conf(ee, conf_level, re_sample, trajectories_num, seed=1)

    assert absolute_result == expected_result
