#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np

def gen_golden_data_simple():
    input_x = np.random.uniform(0, 1, [8, 2048]).astype(np.float16)
    input_y = np.random.uniform(0, 1, [8, 2048]).astype(np.float16)
    input_x[0][0] = 2.871
    # golden = (input_x + input_y).astype(np.float16)
    golden = np.sinh(input_x)
    print(input_x)
    print(golden)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
