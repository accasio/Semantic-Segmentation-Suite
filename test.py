import os, time, cv2, sys
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True,
                    help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--frontend', type=str, default="ResNet50",
                    help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
dataset_dir = './datasets/'
class_names_list, label_values = helpers.get_label_info(os.path.join(dataset_dir + args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

network, _ = model_builder.build_model(args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, is_training=False,
                                       crop_width=args.crop_width, crop_height=args.crop_height)

# sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights ...')
saver = tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(
    dataset_dir=dataset_dir + args.dataset)

# Create directories if needed
if not os.path.isdir("%s" % "test"):
    os.makedirs("%s" % "test")

if not os.path.isdir("%s/%s" % ("test", args.dataset)):
    os.makedirs("%s/%s" % ("test", args.dataset))

target = open("%s/test_scores.csv" % ("test"), 'w')
target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d" % (ind + 1, len(test_input_names)))
    sys.stdout.flush()

    input_image = np.expand_dims(
        np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]), axis=0) / 255.0
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

    st = time.time()
    output_image = sess.run(network, feed_dict={net_input: input_image})

    run_times_list.append(time.time() - st)

    output_image = np.array(output_image[0, :, :, :])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt,
                                                                                 num_classes=num_classes)

    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou))
    for item in class_accuracies:
        target.write(", %f" % (item))
    target.write("\n")

    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)

    gt = helpers.colour_code_segmentation(gt, label_values)

    cv2.imwrite("%s/%s/%s_pred.png" % ("test", args.dataset, file_name),
                cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s/%s_gt.png" % ("test", args.dataset, file_name), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

target.close()

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)
print("\nAverage test accuracy = ", avg_score)
print("Average per class test accuracies = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("\nAverage precision = ", avg_precision)
print("Average recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average mean IoU score = ", avg_iou)
print("Average run time = ", avg_time)
