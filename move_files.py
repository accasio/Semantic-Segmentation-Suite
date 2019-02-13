from random import seed
seed(1)
from random import shuffle
from os import listdir
from os.path import isfile, join
import os
import numpy as np
from PIL import Image
import glob
import shutil



def names_to_numbers():
    folder = '/home/james/Pictures/720rocsafe'

    print(os.path.join(folder, '/gt/*.png'))
    filelist = glob.glob('/home/james/Pictures/720rocsafe/gt/*.png')

    i = 0
    for file in sorted(filelist):
        os.rename(file, '/home/james/Pictures/720rocsafe/ground_truth/%04d.png' % i)
        i = i + 1


def move_from_file(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            img_name= line.split('regions')[1].split('regions')[0] + '.jpg'
            img_path = '/home/james/bikes/bikes' + img_name
            # os.rename(img_path, '/home/james/Pictures/iccv09Data/train/' + img_name)
            # os.rename(line, '/home/james/Pictures/iccv09Data/train_labels/' + os.path.basename(line))


def create_masks():
    with open('/home/james/Pictures/iccv09Data/val.txt') as train_names:
        for file in train_names:
            file = file.strip('\n')
            filename = os.path.basename(file).split('.regions')[0]
            with open(file) as label_data:
                image = []
                label_data = label_data.read()
                for idx, row in enumerate(label_data.split('\n')):
                    if not row:
                        continue
                    image.append([])
                    for pixel in row.split(' '):
                        new_pix = [0, 0, 0]
                        pixel = int(pixel)
                        if pixel == 0:
                            new_pix = [160, 160, 160]
                        elif pixel == 1:
                            new_pix = [204, 0, 0]
                        elif pixel == 2:
                            new_pix = [255, 153, 51]
                        elif pixel == 3:
                            new_pix = [255, 255, 102]
                        elif pixel == 4:
                            new_pix = [102, 204, 0]
                        elif pixel == 5:
                            new_pix = [0, 153, 153]
                        elif pixel == 6:
                            new_pix = [0, 0, 204]
                        elif pixel == 7:
                            new_pix = [76, 0, 153]

                        image[idx].append(new_pix)

                image = np.array(image)
                im = Image.fromarray(image.astype('uint8'))
                im.save('./iccv09Data/val_labels/%s.png' % filename)
                shutil.copyfile('/home/james/Pictures/iccv09Data/images/' + filename + '.jpg',
                                './iccv09Data/val/%s.jpg' % filename)


def move_train_masks():
    for file in glob.glob('/home/james/Pictures/Images_master/original*.png'):
        filename = os.path.basename(file)
        print(filename)
        shutil.copyfile(file, '/home/james/Pictures/train/images/%s' % filename)


def split_imgs():
    dataset = 'aeroscapes_no_animal'
    my_path = '/home/james/Documents/%s/data/aeroscapes/JPEGImages/' % dataset
    x_list = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    x_list = sorted(x_list)
    print("Images")
    print(x_list[0])
    shuffle(x_list)
    print(x_list[0])

    index_num = int(len(x_list) * 0.8)
    train_cpy = x_list[:index_num]
    test = x_list[index_num:]
    index_num = int(len(train_cpy) * 0.8)
    train = train_cpy[:index_num]
    val = train_cpy[index_num:]

    if not os.path.exists('/home/james/Documents/seg-suite/datasets/%s/val' % dataset):
        os.makedirs('/home/james/Documents/seg-suite/datasets/%s/val' % dataset)
    if not os.path.exists('/home/james/Documents/seg-suite/datasets/%s/test' % dataset):
        os.makedirs('/home/james/Documents/seg-suite/datasets/%s/test' % dataset)
    if not os.path.exists('/home/james/Documents/seg-suite/datasets/%s/train' % dataset):
        os.makedirs('/home/james/Documents/seg-suite/datasets/%s/train' % dataset)

    for file in glob.glob('%s*.jpg' % my_path):
        filename = os.path.basename(file)

        if filename in val:
            shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/%s/val/%s' % (dataset, filename))
        elif filename in test:
            shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/%s/test/%s' % (dataset, filename))
        elif filename in train:
            shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/%s/train/%s' % (dataset, filename))
        else:
            print('%s not found in any lists' % filename)


def split_labels():
    dataset = 'aeroscapes_no_animal'
    my_path = '/home/james/Documents/%s/data/aeroscapes/Visualizations/' % dataset
    x_list = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    x_list = sorted(x_list)
    print("Labels")
    print(x_list[0])
    shuffle(x_list)
    print(x_list[0])

    index_num = int(len(x_list) * 0.8)
    train_cpy = x_list[:index_num]
    test = x_list[index_num:]
    index_num = int(len(train_cpy) * 0.8)
    train = train_cpy[:index_num]
    val = train_cpy[index_num:]

    if not os.path.exists('/home/james/Documents/seg-suite/datasets/%s/val_labels' % dataset):
        os.makedirs('/home/james/Documents/seg-suite/datasets/%s/val_labels' % dataset)
    if not os.path.exists('/home/james/Documents/seg-suite/datasets/%s/test_labels' % dataset):
        os.makedirs('/home/james/Documents/seg-suite/datasets/%s/test_labels' % dataset)
    if not os.path.exists('/home/james/Documents/seg-suite/datasets/%s/train_labels' % dataset):
        os.makedirs('/home/james/Documents/seg-suite/datasets/%s/train_labels' % dataset)


    for file in glob.glob('%s*.png' % my_path):
        filename = os.path.basename(file)

        if filename in val:
            shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/%s/val_labels/%s' % (dataset, filename))
        elif filename in test:
            shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/%s/test_labels/%s' % (dataset, filename))
        elif filename in train:
            shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/%s/train_labels/%s' % (dataset, filename))
        else:
            print('%s not found in any lists' % filename)



def reduce_img_size(base_dir):
    # e.g. base_dir = '/home/james/Documents/seg-suite/datasets/trains/'
    dirs = ['train', 'train_labels', 'test', 'test_labels', 'val_labels', 'val']

    for sub_dir in dirs:
        dir = base_dir + sub_dir

        for file in glob.glob('%s/*.jpg' % dir):
            if '.csv' in file:
                continue
            if os.path.isdir(file):
                continue
            img = Image.open(file)
            img_resized = img.resize((500, 500))
            os.remove(file)
            img_resized.save('%s/%s' % (dir, os.path.basename(file)))
        for file in glob.glob('%s/*.png' % dir):
            if '.csv' in file:
                continue
            if os.path.isdir(file):
                continue
            img = Image.open(file)
            img_resized = img.resize((500, 500))
            os.remove(file)
            img_resized.save('%s/%s' % (dir, os.path.basename(file)))


def double_extension(folder):
    for file in glob.glob('%s/*.png.png' % folder):
        filename = file.split('.png.png')[0]
        os.rename(file, '%s.png' % filename)


def reduce_img_res(base_dir):
    # e.g. base_dir = '/home/james/Documents/seg-suite/datasets/trains/'
    dirs = ['train', 'train_labels', 'test', 'test_labels', 'val_labels', 'val']
    sub_dir = 'val'
    for i in range(10, 110, 10):
        dir = base_dir + sub_dir
        if not os.path.isdir("%s/reresed" % dir):
            os.makedirs("%s/reresed" % dir)

        for file in glob.glob('%s/9.png' % dir):
            img = Image.open(file)
            img.save('%s/reresed/%d-%s' % (dir, i/10, os.path.basename(file)), quality=i)


def append_to_files():

    for file in glob.glob('/home/james/Documents/combinedDataset/dataset/MN/gt/*.png'):
        shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/mNeighbourhood/train_labels/mn-' + os.path.basename(file))

    for file in glob.glob('/home/james/Documents/combinedDataset/dataset/MN/original/*.jpg'):
        shutil.copyfile(file, '/home/james/Documents/seg-suite/datasets/mNeighbourhood/train/mn-' + os.path.basename(file))


def color_converter(find, replace, data):

    r1, g1, b1 = find[0], find[1], find[2] # Original value
    r2, g2, b2 = replace[0], replace[1], replace[2] # Value that we want to replace it with
    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]
    return data


def change_pixels():
    dataset = 'fullTrain'

    dir = '/home/james/Documents/%s/gt/*.png' % dataset

    for file in glob.glob(dir):
        filename = os.path.basename(file)
        im = Image.open(file)
        data = np.array(im)

        data = color_converter([55,181,57], [0,0,0], data)
        data = color_converter([153,108,6], [0,64,0], data)
        data = color_converter([115,176,195], [255,255,0], data)
        data = color_converter([81,13,36], [192,0,0], data)
        data = color_converter([206,190,59], [192,128,0], data)
        data = color_converter([190,225,64], [0,128,0], data)
        data = color_converter([89,121,72], [192,128,128], data)
        data = color_converter([161,171,27], [128,128,0], data)
        data = color_converter([112,105,191], [0,128,128], data)
        data = color_converter([135,169,180], [128,128,128], data)
        data = color_converter([235,208,124], [192,128,0], data)

        im = Image.fromarray(data)
        im.save('/home/james/Documents/seg-suite/datasets/%s/train_labels/%s' % (dataset, filename))


def split_aero():
    with open('/home/james/Documents/combinedDataset/dataset/trn.txt') as file_list:
        content = file_list.readlines()

    for filename in content:
        filename = filename.strip('\n')
        shutil.copyfile('/home/james/Documents/aeroscapes_water/data/aeroscapes/JPEGImages/' + filename + '.jpg',
                        '/home/james/Documents/seg-suite/datasets/aeroscapes/train/'+ filename + '.jpg')
        shutil.copyfile('/home/james/Documents/aeroscapes_water/data/aeroscapes/Visualizations/' + filename + '.png',
                        '/home/james/Documents/seg-suite/datasets/aeroscapes/train_labels/'+ filename + '.png')

    with open('/home/james/Documents/combinedDataset/dataset/test.txt') as file_list:
        content = file_list.readlines()

    for filename in content:
        filename = filename.strip('\n')
        shutil.copyfile('/home/james/Documents/aeroscapes_water/data/aeroscapes/JPEGImages/' + filename + '.jpg',
                        '/home/james/Documents/seg-suite/datasets/aeroscapes/test/'+ filename + '.jpg')
        shutil.copyfile('/home/james/Documents/aeroscapes_water/data/aeroscapes/Visualizations/' + filename + '.png',
                        '/home/james/Documents/seg-suite/datasets/aeroscapes/test_labels/'+ filename + '.png')


    with open('/home/james/Documents/combinedDataset/dataset/val.txt') as file_list:
        content = file_list.readlines()

    for filename in content:
        filename = filename.strip('\n')
        shutil.copyfile('/home/james/Documents/aeroscapes_water/data/aeroscapes/JPEGImages/' + filename + '.jpg',
                        '/home/james/Documents/seg-suite/datasets/aeroscapes/val/'+ filename + '.jpg')
        shutil.copyfile('/home/james/Documents/aeroscapes_water/data/aeroscapes/Visualizations/' + filename + '.png',
                        '/home/james/Documents/seg-suite/datasets/aeroscapes/val_labels/'+ filename + '.png')


if __name__ == '__main__':
    # reduce_img_size('/home/james/Documents/seg-suite/datasets/rocsafe/')
    append_to_files()