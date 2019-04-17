import os
from functools import partial, reduce
from typing import Iterable

import numpy
import pandas
import scipy

# import constants

def debug_print(*args, level=1, **kwargs):
    if level<3:
        print(" "*2*level, ends='')
        print(*args, **kwargs)

def simple_test():
    lists = [[1, 2, 3, 4, 5, 6],
             [2, 7, 1, 9, 7]]
    result = reduce(numpy.intersect1d, lists) if lists else []
    print(list(result))

###############################################################
############## PANDAS SPECIFIC FUNCTIONS ######################
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

def __pandas_field_intersection(field_name, *datasets) -> Iterable:
    """From each dataset get unique values from FIELD_NAME
    and return their intersection"""

    lists = map(partial(__pandas_get_unique_values, field_name),
                datasets)
    return reduce(numpy.intersect1d, lists) if datasets else []

def __pandas_strip_columns(dataset, columns):
    useless = columns ^ set(dataset.columns)
    debug_print("Deleting from dataset {}".format(columns), level=2)

def __pandas_strip_references(dataset):
    pass

def __pandas_only_users(dataset, user_ids):
    pass
###############################################################

def _script_relative(relative_path):
    script_path = os.path.realpath(__file__)
    return os.path.join(os.path.dirname(script_path),
                        relative_path)

def hash_params(*args, **kwargs):
    """Return hash of all arguments.  Hash is same for
    different order of arguments"""
    # Helper function
    def hash_recursive(arg_iterable, start=0):
        if isinstance(arg_iterable, dict): # In dict, hash tuples
            return hash_recursive(arg_iterable.items(), start)
        else:
            try: # Try if it is iterable
                iterator = iter(arg_iterable)
                item = next(iterator)
                if item == arg_iterable:
                    raise TypeError # Infinite cycle
                # Update value
                start = hash_recursive(item, start)
                return hash_recursive(iterator, start) # Tail recursion
            except TypeError: 
                return hash(arg_iterable) + start
            except StopIteration:
                return start
    #################
    return hash_recursive([args, kwargs])


def get_trivago_datasets(columns, percentage=1, seed=1):
    """"""
    folder_name = 'data'
    columns = set(columns + ["user_id", "reference"])
    # Compute names
    file_name = str(hash_params(columns, percentage, seed))
    # Check for existence
    os.makedirs(_script_relative(folder_name), exist_ok=True)
    dataset_path = _script_relative(folder_name + file_name)

    if not os.path.exists(dataset_path):
        # Create dataset
        train_path = _script_relative('trivago/train.csv')
        test_path = _script_relative('trivago/test.csv')

        train = __pandas_get_dataset(train_path)
        train = __pandas_strip_columns(train, columns)
        train = __pandas_strip_dataset(train, fields=columns,
                                       percentage=percentage, seed=seed)

        test = __pandas_get_dataset(test_path)
        # Save dataset
        train.to_pickle(path=train_path)
        test.to_pickle(path=test_path)
    else: # Load dataset
        pass


    # Return

if __name__ == "__main__":
    # Load datasets
    get_trivago_datasets([], percentage=0.1)

    # print(len(__pandas_field_intersection('user_id', test, test)))

