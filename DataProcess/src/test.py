import numpy as np
import pickle
import cv2
import os

def test():
    str_set = set()
    str_set.add('apple')
    str_set.add('banana')
    str_set.add('apple')
    str_set.add('grape')
    print('str_set:', str_set)


def test2():
    pklpath = r'E:\School\Grad1\CAD\DeepLearn\Wall\VecFloorSeg\Datasets\R2V\merge_test_phase_V4.pkl'
    try:
        with open(pklpath, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"错误: 文件 {pklpath} 未找到.")
    except pickle.UnpicklingError:
        print("错误: 文件不是有效的pickle文件.")
    except Exception as e:
        print(f"发生了一个错误: {e}")
    # print('data:', data)
    keys = data.keys()
    print('keys:', keys)


if __name__ == '__main__':
    test()