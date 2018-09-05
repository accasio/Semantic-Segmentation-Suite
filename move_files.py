from random import seed
seed(1)
import os
import numpy as np
from PIL import Image
import glob
import shutil

from random import shuffle



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
    x_list = [str(x) for x in range(1,1000)]
    shuffle(x_list)
    index_num = int(len(x_list) * 0.8)
    train_cpy = x_list[:index_num]
    test = x_list[index_num:]
    index_num = int(len(train_cpy) * 0.8)
    train = train_cpy[:index_num]
    val = train_cpy[index_num:]

    for file in glob.glob('/home/james/Pictures/train/images/*.png'):
        filename = os.path.basename(file)
        filename = filename.split('original_')[1].split('.png')[0]
        if filename in val:
            shutil.copyfile(file, '/home/james/Pictures/train/val/%s' % filename)
        elif filename in test:
            shutil.copyfile(file, '/home/james/Pictures/train/test/%s' % filename)
        elif filename in train:
            shutil.copyfile(file, '/home/james/Pictures/train/train/%s' % filename)
        else:
            print('%s not found in any lists' % filename)


def split_labels():
    x_list = [str(x) for x in range(1,1000)]
    shuffle(x_list)
    index_num = int(len(x_list) * 0.8)
    train_cpy = x_list[:index_num]
    test = x_list[index_num:]
    index_num = int(len(train_cpy) * 0.8)
    train = train_cpy[:index_num]
    val = train_cpy[index_num:]

    for file in glob.glob('/home/james/Pictures/train/labels/*.png'):
        filename = os.path.basename(file)
        filename = filename.split('segmented_train_')[1].split('.png')[0]
        if filename in val:
            shutil.copyfile(file, '/home/james/Pictures/train/val_labels/%s.png' % filename)
        elif filename in test:
            shutil.copyfile(file, '/home/james/Pictures/train/test_labels/%s.png' % filename)
        elif filename in train:
            shutil.copyfile(file, '/home/james/Pictures/train/train_labels/%s.png' % filename)
        else:
            print('%s not found in any lists' % filename)



if __name__ == '__main__':
    import tensorflow as tf

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./checkpoints/trains/latest_model_DeepLabV3_trains.ckpt.meta')
        saver.restore(sess, "./checkpoints/trains/checkpoint")
    # split_labels()
