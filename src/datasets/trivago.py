import os
import random
from functools import partial, reduce
from typing import Iterable

import numpy
import pandas
import scipy

# import constants

REL_TRAIN_PATH = "data/trivago_train.csv"
REL_TEST_PATH = "data/trivago_test.csv"
USER_ID = "user_id"
CACHE_FOLDER = 'data'
COLUMNS = [USER_ID, "reference"]
ACTIONS = ["clickout item", "interaction item image"]


def debug_print(*args, level=1, **kwargs):
    if level<9:
        print(" "*2*level, end='')
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
    debug_print("Deleting from dataset {}".format(useless), level=2)
    debug_print("Dataset shape before {}".format(dataset.shape), level=3)
    dataset = dataset.drop(useless, axis=1)
    debug_print("Dataset shape after {}".format(dataset.shape), level=3)
    return dataset

def __pandas_trivago_invalid_rows(dataset):
    action_type = "action_type"
    assert action_type in dataset
    debug_print("Preprocessing trivago", level=2)
    debug_print("Dataset shape before {}".format(dataset.shape), level=3)
    dataset = dataset[dataset[action_type].isin(ACTIONS)].dropna()
    debug_print("Dataset shape after {}".format(dataset.shape), level=3)
    return dataset

def __pandas_trivago_drop_unique(dataset, column, percentage=0):
    assert percentage >= 0 and percentage <= 1
    unique_values = list(dataset[column].unique())
    debug_print("Droping users, size before {} ({}%)".format(len(dataset),
                                                             percentage*100), level=2)
    debug_print("Unique users, before {}".format(len(unique_values)), level=3)

    drop_num = int(len(unique_values) * percentage)
    drop_values = random.sample(unique_values, drop_num)
    result = dataset[~dataset[column].isin(drop_values)]

    debug_print("Unique users, after {}".format(len(result[column].unique())), level=3)
    debug_print("Droping users, size after {}".format(len(result)), level=2)
    return result

def __pandas_drop_top(dataset, column, percentage=0):
    debug_print("Dropping top users {}%".format(percentage*100), level=2)
    dataset['count'] = dataset.groupby(column).transform('count')
    dataset.sort_values('count', inplace=True, ascending=False)
    unique_users = dataset[column].unique()
    drop_users = unique_users[:int(len(unique_users)*percentage)]
    debug_print("Count {}".format(len(drop_users)), level=3)
    debug_print("Largest before {}".format(list(dataset['count'])[0]), level=3)
    dataset = dataset[~dataset[column].isin(drop_users)]
    dataset = dataset[dataset['count'] > 4]
    debug_print("Largest after {}".format(list(dataset['count'])[0]), level=3)
    del dataset['count']
    return dataset

def __pandas_trivago_plot_density(dataset, column, filename=None):
    data = dataset[column].value_counts()
    import matplotlib.pyplot as plt
    plt.hist(data)
    # plt.ylim(top=100)
    # plt.xlim(left=30, right=1400)
    plt.show()

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


def get_trivago_datasets(columns, percentage=1, seed=1,
                         uitems_min=2, uitem_max=-1):
    """Doc
        - uitem_min - minimum number of interaction for user
    """
    debug_print("Loading trivago datasets", level=0)
    columns = set(columns + COLUMNS)
    # Compute names
    file_name = str(hash_params(columns, percentage, seed))
    # Check for existence
    os.makedirs(_script_relative(CACHE_FOLDER), exist_ok=True)
    dataset_path = _script_relative(CACHE_FOLDER + file_name)

    # Check cached
    if not os.path.exists(dataset_path):
        # Create dataset
        train_path = _script_relative(REL_TRAIN_PATH)
        test_path = _script_relative(REL_TEST_PATH)

        train = __pandas_get_dataset(train_path)
        train = __pandas_trivago_invalid_rows(train)
        train = __pandas_strip_columns(train, columns)
        __pandas_trivago_plot_density(train, USER_ID)
        train = __pandas_trivago_drop_unique(train, USER_ID, percentage=0)
        train = __pandas_drop_top(train, USER_ID, percentage=0.03)

        test = __pandas_get_dataset(test_path)
        test = __pandas_trivago_invalid_rows(test)
        test = __pandas_strip_columns(test, columns)
        test = test[~test[USER_ID].isin(train[USER_ID].unique())]
        # Save dataset
        #train.to_pickle(path=train_path)
        #test.to_pickle(path=test_path)
    else: # Load dataset
        pass


    # Return

if __name__ == "__main__":
    # Load datasets
    get_trivago_datasets([], percentage=0.1)

    # print(len(__pandas_field_intersection('user_id', test, test)))

def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'
