import os
import numpy as np
from PIL import Image
import glob
import shutil


def move_from_file(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip('\n')
            img_name= line.split('regions')[1].split('regions')[0] + '.jpg'
            img_path = '/home/james/bikes/bikes' + img_name
            # os.rename(img_path, '/home/james/Pictures/iccv09Data/train/' + img_name)
            # os.rename(line, '/home/james/Pictures/iccv09Data/train_labels/' + os.path.basename(line))


def create_masks():
    with open('/home/james/Pictures/iccv09Data/train.txt') as train_names:
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
                im.save('./iccv09Data/train_labels/%s.png' % filename)
                shutil.copyfile('/home/james/Pictures/iccv09Data/images/' + filename + '.jpg',
                                './iccv09Data/train/%s.jpg' % filename)
if __name__ == '__main__':
    create_masks()
