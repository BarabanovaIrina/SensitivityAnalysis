import pandas as pd
from os.path import join
from vars_executor import read_factor_space, read_vars_inp


def test_read_factor_space():
    path = '../'
    df = pd.read_csv(join(path, 'FactorSpace.csv'), encoding='latin-1')
    lst = list()
    for index in range(df.shape[0]):
        factor = [df.loc[index, 'Name'], df.loc[index, 'LB'], df.loc[index, 'UB']]
        lst.append(factor)

    original_output = read_factor_space(path)

    assert len(lst) == len(original_output)
    assert df.loc[0, 'Name'] == original_output[0].name


def test_vars_inp():
    filename = 'VARS_inp.txt'
    path = '../'
    with open(join(path, filename), 'r', encoding='latin-1') as file:
        raw_text = file.readlines()
    smpstrtgy = raw_text[11]

    original_output = read_vars_inp(path)
    ivars = [float(i) for i in raw_text[7].split()]
    assert smpstrtgy == original_output['SmpStrtgy']
    assert ivars == original_output['IVARS']
