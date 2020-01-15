from abc import ABC
import pandas as pd
import numpy as np


class ModelInputFactor(ABC):

    def __init__(self, name: str, low_bound: float, upper_bound: float):
        self.name = name
        self.low_bound = low_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.name!r}: [{self.low_bound!r}, {self.upper_bound!r}])'


class Container:

    def __init__(self, factors_list: list):
        self.names = np.array([factors_list[i].name for i in range(len(factors_list))])
        self.l_bounds = np.array([factors_list[i].low_bound for i in range(len(factors_list))])
        self.u_bounds = np.array([factors_list[i].upper_bound for i in range(len(factors_list))])
        self.size = len(self.names)


def read_input_file(file: str) -> tuple:
    df = pd.read_csv(file, encoding='latin-1')
    names = np.array(df['Name'])
    lower_bounds = np.array(df['LB'])
    upper_bounds = np.array(df['UB'])
    return names, lower_bounds, upper_bounds


def make_input_factors_container(names: np.array, l_bounds: np.array, up_bounds: np.array) -> list:
    result = list()
    for i in range(len(names)):
        factor = ModelInputFactor(names[i], l_bounds[i], up_bounds[i])
        result.append(factor)

    return result
