import pandas as pd
import numpy as np
from inputfactors import read_input_file, ModelInputFactor, make_input_factors_container
import os.path


def test_read_input_file():
    test_file_path = str(os.path.dirname(__file__))
    filename = 'test_file.csv'
    test_file = os.path.join(test_file_path, filename)
    df = pd.read_csv(test_file)
    expected_result = df['Factor']
    absolute_result, _, _ = read_input_file(test_file)

    assert absolute_result.all() == expected_result.all()


def test_make_input_factors_container():
    test_file_path = str(os.path.dirname(__file__))
    filename = 'test_file.csv'
    test_file = os.path.join(test_file_path, filename)
    df = pd.read_csv(test_file)
    test_container = list()
    for i in range(len(df['Factor'])):
        factor = ModelInputFactor(df['Name'][i], df['LB'][i], df['UB'][i])
        test_container.append(factor)

    actual_result = make_input_factors_container(np.array(df['Name']), np.array(df['LB']), np.array(df['UB']))

    assert actual_result[0].low_bound == test_container[0].low_bound
