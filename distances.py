import numpy as np


def hamming_distance(left_descriptor, right_descriptor):
    left_xor = np.int64(np.bitwise_xor(np.int64(left_descriptor), right_descriptor))
    left_distance = np.zeros(shape=(left_descriptor.shape[0], left_descriptor.shape[1]), dtype=np.uint32)
    while not np.all(left_xor == 0):
        tmp = left_xor - 1
        mask = left_xor != 0
        left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
        left_distance[mask] = left_distance[mask] + 1
    return left_distance


def l2_distance(left_descriptor, right_descriptor):
    return np.sum((left_descriptor - right_descriptor) ** 2.0, axis=2)
