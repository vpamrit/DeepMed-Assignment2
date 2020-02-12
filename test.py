import sys
import os
import argparse
import model
import torch
import torchvision
import skimage
import re
import PIL
import densenet as densemodel

from skimage import io
from os.path import join, isfile
from os import listdir
from PIL import Image
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

    if argv.image_dir != '':
        mdata = SkinDataset(argv.labels_file, argv.image_dir)

    for f in files:
        if not torch.cuda.is_available():
            print("The model needs to be loaded to a GPU")

        device = torch.device('cuda')

        net = model.ResNet50(num_classes=2).to('cuda')
        net.eval()

        with torch.no_grad():

            net.load_state_dict(torch.load(argv.model_path))
            raw_img = Image.open(f)
            image = torchvision.transforms.functional.to_tensor(raw_img)

            raw_pred = net(image.unsqueeze(0).float().to('cuda'))
            pred_labels = torch.nn.functional.softmax(raw_pred, dim=1).tolist()[0]
            print(pred_labels)

            prediction = pred_labels.index(max(pred_labels))
            print("Predicted {}".format(prediction))
            print(type(prediction))

            if argv.labels_file != '':
                actual = get_label(f, mdata)
                print("Actual {}".format(actual))
                total += 1
                result = int(actual) == prediction
                total_correct += result

                correct_count[int(actual)] += result
                total_count[int(actual)] += 1

            print(f)
            #print("X: {0:.4f}".format(pred_labels[0]))
            #print("Y: {0:.4f}".format(pred_labels[1]))

    if argv.image_dir != '' and argv.labels_file != '':
        print("ACCURACY: {}".format(float(total_correct / total)))
        for i in range(len(total_count)):
            print("CATEGORY {} ACCURACY {}".format(i, float(correct_count[i]/total_count[i])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #single image outputs
    parser.add_argument('--image_path', type=str, default='./data/test/120.jpg', help='test image directory')
    parser.add_argument('--model_path', type=str, default='./models/resnet32.pt' , help='path for model to load')
    parser.add_argument('--labels_file', type=str , default='', help='labels file for visualized images')
    parser.add_argument('--binary_mode', type=str , default='', help='put in binary mode')

    #args for directory-based outputs
    parser.add_argument('--image_dir', type=str, default='', help='image directory if provided')

    args = parser.parse_args()
    print(args)
    main(args)

