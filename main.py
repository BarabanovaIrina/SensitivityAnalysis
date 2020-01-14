from SALib.test_functions import Ishigami


import sample
import os.path
from inputfactors import read_input_file, ModelInputFactors
from analysis import morris


if __name__ == '__main__':
    file_name = 'i_factor_space.csv'
    file_name = os.path.join('factorSpace', file_name)
    samples_number = 100
    f_names, f_low_bounds, f_up_bounds = read_input_file(file_name)
    model_input_factors = ModelInputFactors(f_names, f_low_bounds, f_up_bounds)
    sample_to_go = sample.latin_sample(samples_number, model_input_factors)
    model_output = Ishigami.evaluate(sample_to_go)

    analysis_output = morris(model_input_factors, sample_to_go, model_output, resamples=1000, conf_level=0.95)

    print(analysis_output)

