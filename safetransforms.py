import torch
import math
import torchvision
import numpy
import PIL
import random
from PIL import Image
from numpy import array
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.transforms import functional as func

IMG_WIDTH = 600
IMG_HEIGHT = 450
MIN_WIDTH = int(IMG_WIDTH/2.5)
MIN_HEIGHT = int(IMG_HEIGHT/2.5)

def happened(prob):
    if prob == 1:
        return True

    return random.randint(1, 1000) < 1000*prob

class SafeRotate(object):
    def __init__(self, prob):
        self.starting_pos = [0, 90, 180, 270]
        self.minwidth=MIN_WIDTH
        self.minheight=MIN_HEIGHT
        self.ratio = IMG_HEIGHT/IMG_WIDTH #H/W
        self.prob = prob

        return

    #computes coordinates of region after rotation based on the original image
    def cropp_rotated(self, image, degrees):
        x, y = image.size
        cosA = abs(math.cos(math.radians(degrees)))
        sinA = abs(math.sin(math.radians(degrees)))

        a = x * cosA
        b = x * sinA

        relation = a / (a + b)
        right_indent1 = a - x * relation * cosA

        relation = b / (a + b)
        bottom_ident1 = b - x * relation *sinA


        c = y * cosA
        d = y * sinA

        relation = c / (c + d)
        right_indent2 = c - y * relation * cosA

        relation = d / (c + d)
        bottom_ident2 = d - y * relation *sinA

        right_indent = max(right_indent1, right_indent2)
        top_indent = max(bottom_ident1, bottom_ident2)

        #size of rotated image:
        w_rotated = x * cosA + y * sinA
        h_rotated = y * cosA + x * sinA


        box = [
        int(top_indent),
        int(right_indent),
        int(h_rotated - top_indent)+1-top_indent,
        int(w_rotated - right_indent)-right_indent
        ]

        return box


    def __call__(self, x):

        if not happened(self.prob):
            return x

        start_pos = self.starting_pos[random.randint(0, 3)]

        x = torchvision.transforms.functional.rotate(x, start_pos, expand=True)

        if start_pos == 90 or start_pos == 270:
            crop_width = IMG_HEIGHT
            crop_height = int(IMG_HEIGHT * self.ratio)
            yfin = int(IMG_WIDTH/2 - crop_height/2)
            xfin = 0
            x = torchvision.transforms.functional.crop(x, yfin, xfin, crop_height, crop_width)


        possible_angles = [i for i in range(-30, 30, 5)]
        final_angle = possible_angles[random.randint(0, len(possible_angles)-1)]

        #perform the final operations
        box = self.cropp_rotated(x, final_angle)

        x = torchvision.transforms.functional.rotate(x, final_angle, expand=True, resample=False)
        x = torchvision.transforms.functional.resized_crop(x, box[0], box[1], box[2], box[3], (IMG_HEIGHT, IMG_WIDTH))


        return x
