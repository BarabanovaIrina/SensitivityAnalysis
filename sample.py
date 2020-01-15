import numpy as np

from inputfactors import Container


def scale_sample(sample, l_bounds, u_bounds):
    """
    Generate scaled sample within real problem bounds

    """
    scaled_sample = (u_bounds-l_bounds) * sample
    scaled_sample += l_bounds
    return scaled_sample


def lhc_sample(number_of_samples: int, input_params: Container, seed=None):
    """
    Generate a set of samples with Latin Hypercube sample method

    :param number_of_samples:
    :param input_params: class example with all parameters' names and bounds
    :param seed:
    :return: NumPy array of size [samples_number, parameters_number]
    """

    if seed:
        np.random.seed(seed)

    number_of_params: int = input_params.size
    result: np.array = np.zeros([number_of_samples, number_of_params])
    temporary_result = np.zeros([number_of_samples])
    levels_num = 1.0 / number_of_samples

    for i in range(number_of_params):
        for j in range(number_of_samples):
            temporary_result[j] = np.random.uniform(low=j*levels_num, high=(j+1)*levels_num)

        np.random.shuffle(temporary_result)
        result[:, i] = temporary_result

    scaled_result = scale_sample(result, input_params.l_bounds, input_params.u_bounds)

    return scaled_result
