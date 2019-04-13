import os
from functools import partial, reduce
from typing import Iterable

import numpy
import pandas
import scipy

# import constants

def debug_print(*args, level=1, **kwargs):
    if True:
        print(*args, **kwargs)

def simple_test():
    lists = [[1, 2, 3, 4, 5, 6],
             [2, 7, 1, 9, 7]]
    result = reduce(_reduce_intersect, lists) if lists else []
    print(list(result))


def __pandas_get_dataset(filename) -> pandas.DataFrame:
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    debug_print("Loading dataset {}".format(filename))
    dataset = pandas.read_csv(filename)

    debug_print("Length: {:,} x {:,}".format(dataset.shape[0], dataset.shape[1]))
    debug_print("Unique user_id count: {:,}".format(*dataset['user_id'].unique().shape), level=2)
    return dataset

def __pandas_get_unique_values(field_name,
                               dataset: pandas.DataFrame) -> pandas.Series:
    return dataset[field_name].unique()


def _script_relative(relative_path):
    script_path = os.path.realpath(__file__)
    return os.path.join(os.path.dirname(script_path),
                        relative_path)

def _get_intersect(list1, list2) -> list:
    return numpy.intersect1d(list1, list2)

def _get_field_intersection(field_name, *datasets) -> Iterable:
    """From each dataset get unique values from FIELD_NAME
    and return their intersection"""

    lists = map(partial(__pandas_get_unique_values, field_name),
                datasets)
    return reduce(_get_intersect, lists) if datasets else []


if __name__ == "__main__":
    # Load datasets
    simple_test()
    train_path = _script_relative('trivago/train.csv')
    test_path = _script_relative('trivago/test.csv')
    train = __pandas_get_dataset(train_path)
    test = __pandas_get_dataset(test_path)

    print(len(_get_field_intersection('user_id', train, test)))
