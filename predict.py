import os,time,cv2, sys, math
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=0, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=0, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--frontend', type=str, default="ResNet50",
                    help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

dataset_dir = './datasets/'

class_names_list, label_values = helpers.get_label_info(os.path.join(dataset_dir + args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

args.image = '/home/james/Pictures/aerial/' + 'pier.jpg'

loaded_image = utils.load_image(args.image)

if args.crop_height == 0 and args.crop_width == 0:
    height, width, _ = loaded_image.shape
    args.crop_height = height
    args.crop_width = width

network, _ = model_builder.build_model(args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, is_training=False, crop_height=args.crop_height, crop_width=args.crop_width)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')

saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

print("Testing image " + args.image)

resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

st = time.time()
output_image = sess.run(network,feed_dict={net_input:input_image})

run_time = time.time()-st

output_image = np.array(output_image[0,:,:,:])
output_image = helpers.reverse_one_hot(output_image)

out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
file_name = utils.filepath_to_name(args.image)
if not os.path.isdir("./predicted/"):
    os.makedirs("./predicted/")
if not os.path.isdir("./predicted/%s" % args.dataset):
    os.makedirs("./predicted/%s" % args.dataset)
cv2.imwrite("./predicted/%s/%s.png" % (args.dataset, file_name), cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
cv2.imwrite("./predicted/%s/%s_pred.png" % (args.dataset, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

print("")
print("Finished!")
print("Wrote image " + "./predicted/%s/%s_pred.png" % (args.dataset, file_name))