import pandas as pd
import numpy as np
from inputfactors import read_input_file, ModelInputFactor, make_input_factors_container
import os.path


def test_read_input_file():
    filename = 'test_file.csv'
    df = pd.read_csv(filename)
    expected_result = df['Factor']
    file = os.path.join('../tests', filename)
    print(file)
    absolute_result, _, _ = read_input_file(file)

    assert absolute_result.all() == expected_result.all()


def test_make_input_factors_container():
    filename = 'test_file.csv'
    df = pd.read_csv(filename)
    container = list()
    for i in range(len(df['Factor'])):
        factor = ModelInputFactor(df['Name'][i], df['LB'][i], df['UB'][i])
        container.append(factor)

    actual_result = make_input_factors_container(np.array(df['Name']), np.array(df['LB']), np.array(df['UB']))

    assert actual_result[0].low_bound == container[0].low_bound
