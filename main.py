import sample
import os.path
import numpy as np
from SALib.test_functions import Ishigami
from inputfactors import (read_input_file,
                          make_input_factors_container,
                          Container,
                          )
from analysis import morris


if __name__ == '__main__':
    file_name = 'i_factor_space.csv'
    file_name = os.path.join('factorSpace', file_name)
    samples_number = 100

    f_names: np.array
    f_low_bounds: np.array
    f_low_bounds: np.array

    f_names, f_low_bounds, f_up_bounds = read_input_file(file_name)
    factors_container: list = make_input_factors_container(f_names, f_low_bounds, f_up_bounds)
    model_input_factors = Container(factors_container)
    sample_to_go = sample.lhc_sample(samples_number, model_input_factors)
    model_output = Ishigami.evaluate(sample_to_go)

    analysis_output = morris(model_input_factors, sample_to_go, model_output, resamples=1000, conf_level=0.95)

    print(analysis_output)
