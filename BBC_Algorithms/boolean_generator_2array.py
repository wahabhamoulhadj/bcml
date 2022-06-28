import numpy as np


def bf_output(key, arr1, arr2):
    key += 1
    if key == 1:
        return np.logical_and(arr1, arr2) * 1
    elif key == 2:
        return np.logical_and(1 - arr1, arr2) * 1
    elif key == 3:
        return np.logical_and(arr1, 1 - arr2) * 1
    elif key == 4:
        return 1 - (np.logical_and(arr1, arr2) * 1)
    elif key == 5:
        return np.logical_or(arr1, arr2) * 1
    elif key == 6:
        return np.logical_or(1 - arr1, arr2) * 1
    elif key == 7:
        return np.logical_or(arr1, 1 - arr2) * 1
    elif key == 8:
        return 1 - (np.logical_or(arr1, arr2) * 1)
    elif key == 9:
        return np.logical_xor(arr1, arr2) * 1
    elif key == 10:
        return 1 - (np.logical_xor(arr1, arr2) * 1)
