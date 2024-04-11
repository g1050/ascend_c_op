#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
def addcmul(tensor1, tensor2, tensor3, value):
    result = tensor2 + value * (tensor1 * tensor3)
    return result

def gen_golden_data_simple():
    input_x = np.random.uniform(1, 2, [8, 2048]).astype(np.float16)
    input_y = np.random.uniform(1, 2, [8, 2048]).astype(np.float16)
    golden = addcmul(input_x,input_x,input_y,input_y)
    print(input_x[0][0],input_x[0][0],input_y[0][0],input_y[0][0])
    print(golden[0][0])
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

# ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
# ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);
# ReadFile("./input/input_x.bin", inputByteSize, input_data, inputByteSize);
# ReadFile("./input/input_y.bin", inputByteSize, value, inputByteSize);
if __name__ == "__main__":
    gen_golden_data_simple()
