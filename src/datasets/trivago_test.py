import pytest
from datasets.trivago import *



def test_hashing():
    assert hash_params([]) == hash_params([])
    assert hash_params((1, 2, 3)) != hash_params()
    assert hash_params((1, 2, 3)) == hash_params((1, 2, 3))
    hash1 = hash_params({"a":1, "k":"m"}, "aa")
    hash2 = hash_params({"a":1, "k":"m"}, "aa")
    assert hash1 == hash2
