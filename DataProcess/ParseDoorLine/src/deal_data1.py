import os
import shutil

def doSelect1():
    pdfpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\pdfs'
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\images'
    outpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\pdfs'
    os.makedirs(outpath, exist_ok=True)

    pdfs = os.listdir(pdfpath)
    pdfs = [os.path.splitext(pdf)[0] for pdf in pdfs]
    imgs = os.listdir(imgpath)
    imgs = [os.path.splitext(img)[0] for img in imgs]
    pdfs_new = [pdf for pdf in pdfs if pdf in imgs]
    print('total:', len(pdfs_new))
    for pdf in pdfs_new:
        shutil.copy(os.path.join(pdfpath, pdf) + '.pdf', outpath)
    print('----- finish -----')

def doRename():
    imgpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-ori\images'
    labelpath = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-ori\labels'
    imgout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-rename\images'
    labelout = r'E:\School\Grad1\CAD\Datasets\DwgFiles\DoorLineData\dataset1-pdf\datasets\test3\dataset-select\data-aug\data-aug-rename\labels'
    os.makedirs(imgout, exist_ok=True)
    os.makedirs(labelout, exist_ok=True)

    imgs = os.listdir(imgpath)
    for i, img in enumerate(imgs):
        if i % 100 == 0:
            print('%d / %d' % (i, len(imgs)))
        label = img.replace('.jpg', '.txt')
        img_new = 'img_' + str(i + 1)
        shutil.copy(os.path.join(imgpath, img), os.path.join(imgout, img_new + '.jpg'))
        shutil.copy(os.path.join(labelpath, label), os.path.join(labelout, img_new + '.txt'))
    print('finish')


if __name__ == '__main__':
    doRename()
