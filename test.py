import sys
import os
import argparse
import torch
import torchvision
import skimage
import re
import PIL
import densenet as densenetmodel
import resnet as resnetmodel
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import color_constancy as cc

from sklearn.metrics import confusion_matrix
from skimage import io
from os.path import join, isfile
from os import listdir
from PIL import Image
from torchvision import transforms
from load_data import SkinDataset

def get_files(argv):
    if argv.image_dir != '':
        root_dir = argv.image_dir
        files = [root_dir+f for f in os.listdir(root_dir) if isfile(join(root_dir, f))]
    else:
        files = [argv.image_path]

    return files

def get_label(f, mdata):
    filename = re.split('[/.]', f)[-2]

    return mdata.get_label_by_img_name(filename)


def main(argv):
    # my code here
    files = get_files(argv)
    mdata = None
    total_correct = 0
    total = 0
    correct_count = [0,0,0,0,0,0,0]
    total_count= [0,0,0,0,0,0,0]
    actuals = []
    predictions = []

    if argv.image_dir != '':
        mdata = SkinDataset(argv.labels_file, argv.image_dir)

    net = densenetmodel.densenet201(target_classes=[0,1,2,3,4,5,6]).to('cuda') # 0 vs others
    net.load_state_dict(torch.load(argv.model_path))
    net2 = None
    net3 = None


    net.eval()

    # we use the 1 vs others and ALL
    # if 1: we use 1 vs 4 and 0 vs 1
    # if not 1: we use Most and 4,others and 0,others combined

    if argv.model_path2 != None:
        net2 = densenetmodel.densenet201(num_classes=7).to('cuda')
        net2.load_state_dict(torch.load(argv.model_path2))
        net2.eval()


    for f in files:
        if not torch.cuda.is_available():
            print("The model needs to be loaded to a GPU")

        device = torch.device('cuda')


        with torch.no_grad():
            raw_image = Image.open(f)

            numpy_image = np.array(raw_image)
            numpy_image = cc.color_constancy(numpy_image)
            raw_image = Image.fromarray(numpy_image)
            raw_image = transforms.functional.resize(raw_image, (225, 300))

            #normalize and transform
            image = transforms.functional.to_tensor(raw_image)
            image = transforms.functional.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

            raw_pred = net(image.unsqueeze(0).float().to('cuda'))
            pred_labels = torch.nn.functional.softmax(raw_pred, dim=1).tolist()[0]


            if argv.model_path2 != None:
                raw_pred2 = net2(image.unsqueeze(0).float().to('cuda'))
                pred_labels2 = torch.nn.functional.softmax(raw_pred2, dim=1).tolist()[0]
                pred_labels = [2*x+y for x,y in zip(pred_labels, pred_labels2)]


            prediction = net.target_classes[pred_labels.index(max(pred_labels))]
            predictions += [prediction]
            print(prediction)

            if argv.labels_file != '':
                actual = get_label(f, mdata)

                if int(actual) in [0,1,2,3,4,5,6]:
                    actuals += [actual]

                    print("Actual {}".format(actual))
                    total += 1
                    result = int(actual) == prediction
                    total_correct += result

                    correct_count[int(actual)] += result
                    total_count[int(actual)] += 1

    if argv.image_dir != '' and argv.labels_file != '':
        print("ACCURACY: {}".format(float(total_correct / total)))
        for i in range(len(total_count)):
            print("CATEGORY {} ACCURACY {}".format(i, float(correct_count[i]/total_count[i])))


        accuracy = sklearn.metrics.accuracy_score(actuals, predictions)
        recall = sklearn.metrics.recall_score(actuals, predictions, average='weighted')
        precision = sklearn.metrics.precision_score(actuals, predictions, average='weighted')

        print("Accuracy {}".format(accuracy))
        print("Recall {}".format(recall))
        print("Precision {}".format(precision))

        target_names=["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        print(len(actuals))
        print(len(predictions))
        cm = confusion_matrix(actuals, predictions)
        print(confusion_matrix)

        # Normalise
        cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show(block=False)
        plt.savefig('./confusion.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #single image outputs
    parser.add_argument('--image_path', type=str, default='./data/test/120.jpg', help='test image directory')
    parser.add_argument('--model_path', type=str, default='./models/densenet8.pt', help='path for model to load')
    parser.add_argument('--model_path2', type=str, default=None, help='path for model to load')
    parser.add_argument('--model_path3', type=str, default=None, help='path for model to load')
    parser.add_argument('--labels_file', type=str , default='', help='labels file for visualized images')
    parser.add_argument('--binary_mode', type=str , default='', help='put in binary mode')

    #args for directory-based outputs
    parser.add_argument('--image_dir', type=str, default='', help='image directory if provided')

    args = parser.parse_args()
    main(args)

