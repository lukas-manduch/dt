import os
import random

from functools import partial, reduce
from hashlib   import md5
from itertools import chain, count
from pickle    import Pickler, Unpickler
from typing    import Iterable

import numpy
import pandas

from scipy.sparse import coo_matrix
# import constants

REL_TRAIN_PATH = "data/trivago_train.csv"
REL_TEST_PATH = "data/trivago_test.csv"
USER_ID = "user_id"
REFERENCE =  "reference"
CACHE_FOLDER = 'cached'
TIMESTAMP = 'timestamp'
COLUMNS = [USER_ID, REFERENCE, TIMESTAMP]
ACTIONS = ["clickout item", "interaction item image"]
ACTION_TYPE = "action_type"


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
def __filename_modif(filename):
    return filename+"mod"

def __pandas_get_dataset(filename) -> pandas.DataFrame:
    filename = __filename_modif(filename)
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    debug_print("Loading dataset {}".format(filename))
    dataset = pandas.read_csv(filename)

    debug_print("Length: {:,} x {:,}".format(dataset.shape[0], dataset.shape[1]))
    debug_print("Unique user_id count: {:,}".format(*dataset['user_id'].unique().shape), level=2)
    return dataset

def __pandas_modify_datasets(*filenames):
    sets = []
    names = []
    debug_print("Reindexing datasets {} ...".format(filenames), level=2)
    for x in filenames:
        if os.path.exists(__filename_modif(x)):
            debug_print("Skip {}".format(x))
            continue
        sets.append(__pandas_strip_columns(pandas.read_csv(x), COLUMNS + [ACTION_TYPE]))
        names.append(x)

    for col in [USER_ID, REFERENCE]:
        debug_print("Reindexing column {} ...".format(col), level=3)
        uniq = set(chain(*map(lambda x: x[col], sets)))
        counter = count()
        mapped = {x: next(counter) for x in uniq}
        for x in sets:
            x[col] = x[col].map(mapped, na_action='ignore')
            import gc
            gc.collect()
    for i in range(len(names)):
        sets[i].to_csv(__filename_modif(names[i]))

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
    useless = set(columns) ^ set(dataset.columns)
    debug_print("Deleting from dataset {}".format(useless), level=2)
    debug_print("Dataset shape before {}".format(dataset.shape), level=3)
    dataset = dataset.drop(useless, axis=1)
    debug_print("Dataset shape after {}".format(dataset.shape), level=3)
    return dataset

def __pandas_trivago_invalid_rows(dataset):
    assert ACTION_TYPE in dataset
    debug_print("Preprocessing trivago", level=2)
    debug_print("Dataset shape before {}".format(dataset.shape), level=3)
    dataset = dataset[dataset[ACTION_TYPE].isin(ACTIONS)]
    debug_print("Dataset shape drop actions {}".format(dataset.shape), level=3)
    dataset = dataset.dropna()
    debug_print("Dataset shape after nan {}".format(dataset.shape), level=3)
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

def __pandas_drop_top(dataset, column, percentage=0, min_items=3):
    """Drop tails
     - min items
     - percentage - Percentage of top users to drop"""
    debug_print("Dropping top users {}%".format(percentage*100), level=2)

    dataset['count'] = dataset.groupby(column).transform('count').iloc[:,1]
    dataset.sort_values('count', inplace=True, ascending=False)
    unique_users = dataset[column].unique()
    drop_users = unique_users[:int(len(unique_users)*percentage)]

    debug_print("Count {}".format(len(drop_users)), level=3)
    debug_print("Largest before {}".format(list(dataset['count'])[0]), level=3)

    dataset = dataset[~dataset[column].isin(drop_users)]
    dataset = dataset[dataset['count'] > min_items]

    debug_print("Largest after {}".format(list(dataset['count'])[0]), level=3)
    del dataset['count']
    return dataset

def __pandas_drop_time(dataset, time=0):
    assert not dataset[TIMESTAMP].empty
    if time <= 0:
        debug_print("Not droppping any time", level=2)
        return dataset
    high = max(dataset[TIMESTAMP])
    debug_print("Dropping times {}".format(time), level=2)
    debug_print("Shape before {}".format(dataset.shape), level=3)
    tmp = dataset[dataset[TIMESTAMP] > high - time]
    debug_print("Shape after {}".format(tmp.shape), level=3)
    return tmp

def __pandas_reindex_values(*datasets, column=''):
    """From 0 in column"""
    debug_print("Reindexing column {} ...".format(column), level=2)
    assert column
    uniq = numpy.unique(list(chain(*map(lambda x: x[column], datasets))))
    counter = count()
    mapped = {x: next(counter) for x in uniq}
    debug_print("Users mapped {}".format(len(mapped)), level=3)
    for x in datasets:
        x[column] = x[column].map(mapped)
    debug_print("Done", level=2)

def __pandas_trivago_plot_density(dataset, column, filename=None):
    data = dataset[column].value_counts()
    import matplotlib.pyplot as plt
    plt.hist(data, bins=20)
    # plt.ylim(top=100)
    plt.yscale('log')
    #plt.xlim(left=3, right=303)
    print(_script_relative("plot"))
    plt.savefig(_script_relative("plot.png"))
    # plt.show()


def __pandas_to_coo(train, test):
    rows = max(max(train[USER_ID]),   max(test[USER_ID])   ) + 1
    cols = max(max(train[REFERENCE]), max(test[REFERENCE]) ) + 1

    train['tmp'] = 1
    test['tmp'] = 1

    matrix1 = coo_matrix((train['tmp'], (train[USER_ID], train[REFERENCE])),
                         shape=(rows, cols))
    matrix2 = coo_matrix((test['tmp'], (test[USER_ID], test[REFERENCE])),
                         shape=(rows, cols))

    del train['tmp']
    del test['tmp']

    return matrix1, matrix2
###############################################################

def _script_relative(relative_path):
    script_path = os.path.realpath(__file__)
    return os.path.join(os.path.dirname(script_path),
                        relative_path)

def hash_params(*args, **kwargs):
    """Return hash of all arguments.  Hash is same for
    different order of arguments"""
    # Helper function
    res = str(args) + str(kwargs)
    m = md5()
    m.update(res.encode('utf-8'))
    return str(m.digest().hex())


def get_trivago_datasets(columns, percentage=1, seed=1,
                         uitems_min=2, lt_drop=0, time=0):
    """Load train and test datasets
        - percentage - how much of train could be dropped
        - uitem_min - minimum number of interaction for user
        - lt_drop - how many % should be dropped from long tail
        - time - how many seconds from last action to keep - 0 = infinite
    """
    debug_print("Loading trivago datasets", level=0)
    columns = set(columns + COLUMNS)
    # Compute names, must be sorted, because sets are kinda random
    file_name = str(hash_params(sorted(columns, reverse=True), percentage, 
                                seed, uitems_min, lt_drop, time))
    # Check for existence
    os.makedirs(_script_relative(CACHE_FOLDER), exist_ok=True)
    dataset_path = _script_relative(os.path.join(CACHE_FOLDER, file_name))

    debug_print("Trying {}".format(dataset_path), level=2)
    # Check cached
    if not os.path.exists(dataset_path):
        debug_print("Not found", level=2)
        # Create dataset
        train_path = _script_relative(REL_TRAIN_PATH)
        test_path = _script_relative(REL_TEST_PATH)

        __pandas_modify_datasets(train_path, test_path)

        train = __pandas_get_dataset(train_path)
        debug_print("Train shape before {}".format(train.shape))

        train = __pandas_strip_columns(train, columns | {ACTION_TYPE})
        train = __pandas_trivago_invalid_rows(train)
        train = __pandas_strip_columns(train, columns)

        train = __pandas_trivago_drop_unique(train, USER_ID, percentage=percentage)
        train = __pandas_drop_top(train, USER_ID, percentage=lt_drop, min_items=uitems_min)
        train = __pandas_drop_time(train, time=time)
        debug_print("Train shape after {}".format(train.shape))

        test = __pandas_get_dataset(test_path)
        test = __pandas_strip_columns(test, columns | {ACTION_TYPE})
        test = __pandas_trivago_invalid_rows(test)
        test = __pandas_strip_columns(test, columns)
        debug_print("Dropping non train {}".format(test.shape), level=2)
        test = test[test[USER_ID].isin(train[USER_ID].unique())]
        debug_print("After non train {}".format(test.shape), level=2)
        # __pandas_reindex_values(train, test, column=USER_ID)
        # __pandas_reindex_values(train, test, column=REFERENCE)

        # Save dataset
        debug_print("Saving dataset {}".format(dataset_path))
        with open(dataset_path, "wb") as f:
            Pickler(f).dump((train, test))
    else: # Load dataset
        with open(dataset_path, "rb") as f:
            debug_print("Found", level=2)
            train, test = Unpickler(f).load()
    print(len(set(test[USER_ID])))
    print(len(set(train[USER_ID]) & set(test[USER_ID])))
    __pandas_trivago_plot_density(train, USER_ID)
    return __pandas_to_coo(train, test)


if __name__ == "__main__":
    print("hi")

