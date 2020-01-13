from abc import ABCMeta
import pandas as pd
import numpy as np


class ModelInputFactors:

    def __init__(self, name: np.array, low_bound: np.array, upper_bound: np.array):
        self.name = name
        self.size = len(self.name)
        self.low_bound = low_bound
        self.upper_bound = upper_bound

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.name!r}: [{self.low_bound!r}, {self.upper_bound!r}])'


def read_input_file(file: str) -> tuple:
    df = pd.read_csv(file, encoding='latin-1')
    names = np.array(df['Factor'])
    lower_bounds = np.array(df['LB'])
    upper_bounds = np.array(df['UB'])
    return names, lower_bounds, upper_bounds




