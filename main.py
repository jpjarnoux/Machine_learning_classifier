#!/usr/bin/env python
# -*-coding:utf8-*-

import numpy as np
import os


def read_file(filepath, delimiter=' '):
    contents = []
    with open(filepath) as f:
        for line in f:
            contents.append(line.strip().split(delimiter))

    return np.array(contents).astype(np.float)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = os.path.abspath("Data")
    print(data_dir)
    train_path = os.path.join(data_dir, "data_tp1_app.txt")
    print(train_path)
    train_data = read_file(train_path)
    test_path = os.path.join(data_dir, "data_tp1_dec.txt")
    test_data = read_file(test_path)



