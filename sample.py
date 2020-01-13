import numpy as np


def scale_sample(sample, l_bounds, u_bounds):
    """
    Generate sample in-place within real problem bounds

    """
    np.add(np.multiply(sample, (u_bounds-l_bounds), out=sample), l_bounds, out=sample)


def latin_sample(number_of_samples: int, input_params, seed=None):
    """
    :param number_of_samples:
    :param input_params: class example with all parameters' names and bounds
    :param seed:
    :return: NumPy array of size [samples_number, parameters_number]
    """

    if seed:
        np.random.seed(seed)

    number_of_params = input_params.size
    result: np.array = np.zeros([number_of_samples, number_of_params])
    temp = np.zeros([number_of_samples])
    d = 1.0/number_of_samples

    for i in range(number_of_params):
        for j in range(number_of_samples):
            temp[j] = np.random.uniform(low=j*d,
                                        high=(j+1)*d, size=1)[0]

        np.random.shuffle(temp)
        for j in range(number_of_samples):
            result[j, i] = temp[j]

    scale_sample(result, input_params.low_bound, input_params.upper_bound)

    return result
