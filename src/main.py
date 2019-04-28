import os
import time
from pickle    import Pickler, Unpickler

import lightfm
import lightfm.evaluation
import numpy

from scipy.sparse import coo_matrix

from datasets import trivago
from datasets.trivago import hash_params, _script_relative


MODELS_FOLDER = 'models'

def get_model(args_dict, train):
    os.makedirs(_script_relative(MODELS_FOLDER), exist_ok=True) 
    path = os.path.join(MODELS_FOLDER, hash_params(args_dict))
    path = _script_relative(path)
    print("Model path {}".format(path))

    if os.path.exists(path):
        print("FOUND cached model ")
        with open(path, "rb") as f:
            return Unpickler(f).load()

    tbef = time.time()
    print("Train start {}".format(tbef))
    model = lightfm.LightFM(loss=args_dict['loss'])
    model.fit(train, epochs=args_dict['epochs'])
    taft = time.time()
    print("Train end {}".format(taft, taft-tbef))

    with open(path, "wb") as f:
        print("Saving model")
        Pickler(f).dump(model)

    return model


def load_normal_half_one_month():
    args = {"percentage" : 0.5,
            "uitems_min" : 4,
            "lt_drop" : 0.1,
            "time" : 3600*24*30,
            }
    train, test = trivago.get_trivago_datasets([], **args)
    args.update(
            {"epochs" : 30,
            "loss"   : 'warp',
            }
            )
    return train, test, get_model(args, train)


def load_normal_half():
    args = {"percentage" : 0.5,
            "uitems_min" : 4,
            "lt_drop" : 0.1,
            }
    train, test = trivago.get_trivago_datasets([], **args)
    args.update(
            {"epochs" : 30,
            "loss"   : 'warp',
            }
            )
    return train, test, get_model(args, train)

def load_normal_half_with_lt():
    args = {"percentage" : 0.5,
            "uitems_min" : 4,
            "lt_drop"    : 0,
            }
    train, test = trivago.get_trivago_datasets([], **args)
    args.update(
                {"epochs" : 30,
                "loss"   : 'warp',
                }
            )
    return train, test, get_model(args, train)

if __name__ == "__main__":
    # train, test, model = load_normal_half_with_lt()
    train, test, model = load_normal_half()
    # train, test, model = load_normal_half_one_month()
    t = coo_matrix(([1 for i in range(0,1000) ], ([1 for i in range(0,1000)],[i for i in range(0,1000)])), shape=test.shape)
    # precision = lightfm.evaluation.precision_at_k(model, test, train_interactions=train)
    precision = lightfm.evaluation.precision_at_k(model, t, train_interactions=train)
    n = numpy.count_nonzero(precision)
    print("Precision is")
    print(sum(precision) / n)


