#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

import numpy as np

def addcmul(tensor1, tensor2, tensor3, value):
    result = tensor2 + value * (tensor1 * tensor3)
    return result

# Example usage
# tensor1 = np.array([1, 2, 3])
# tensor2 = np.array([4, 5, 6])
# tensor3 = np.array([7, 8, 9])
# value = 2

# result = addcmul(tensor1, tensor2, tensor3, value)
# print("Result:", result)


def gen_golden_data_simple():
    x1 = np.random.uniform(1, 2, [8, 2048]).astype(np.float16)
    x2 = np.random.uniform(1, 2, [8, 2048]).astype(np.float16)
    value = np.random.uniform(1, 2, [8, 2048]).astype(np.float16)
    input_dat = np.random.uniform(1, 2, [8, 2048]).astype(np.float16)
    golden = addcmul(x1,input_dat,x2,value)
    print(x1,x2,value,input_dat,golden)
    print(x1[0][0],x2[0][0],value[0][0],input_dat[0][0],golden[0][0])
    x1.tofile("./input/x1.bin")
    x2.tofile("./input/x2.bin")
    value.tofile("./input/value.bin")
    input_dat.tofile("./input/input_dat.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
