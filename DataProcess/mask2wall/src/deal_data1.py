import shutil
import os

def copyData():
    referpath = '../data/images'
    srcpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\labels'
    outpath = '../data/labels_txt'
    os.makedirs(outpath, exist_ok=True)

    files = os.listdir(referpath)
    for f in files:
        fname = os.path.splitext(f)[0]
        shutil.copy(os.path.join(srcpath, fname + '.txt'), outpath)
    print('finish')


if __name__ == '__main__':
    copyData()