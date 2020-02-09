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

#there will be no sample size if batch is (100, 1, 32, 32) => input is
#torchvision.transforms.functional.resized_crop
#torchvision.transforms.functional.rotate
#torchvision.transforms.functional.translate

#sample values
IMG_WIDTH = 490
IMG_HEIGHT = 326
MIN_WIDTH = int(IMG_WIDTH/2.5)
MIN_HEIGHT = int(IMG_HEIGHT/2.5)

#where prob is just a probability between 0 and 1 as a float
def happened(prob):
    if prob == 1:
        return True

    return random.randint(1, 1000) < 1000*prob

#assumes channel 0 is the only channel
def findNewLabel(ten):
    m = ten[0].view(1, -1).argmax(1)

    #used to be IMG_WIDTH here for ten.size()
    tmax0 = torch.max(ten[0])
    tmax1 = torch.max(ten[1])
    tmax2 = torch.max(ten[2])
    indices = torch.cat(((m / ten.size()[2]).view(-1, 1), (m % ten.size()[2]).view(-1, 1)), dim=1).float()
    x = indices[0,1].item()
    y = indices[0,0].item()

    indices[0,0] = float(x)/ten.size()[2]
    indices[0,1] = float(y)/ten.size()[1]

    return indices.numpy()

#takes the pixel-weighted average of coordinates to compute new label
def calculateLabel(pil):
    avgIndex = [0,0]
    total = 0

    width, height = pil.size

    for i in range(width):
        for j in range(height):
            r = pil.getpixel((i,j))[0]
            if r > 0:
                total += r


    for i in range(width):
        for j in range(height):
            r = pil.getpixel((i,j))[0]
            if r > 0:
                avgIndex[0] += r*i / total
                avgIndex[1] += r*j / total

    avgIndex[0] /= width
    avgIndex[1] /= height

    avgIndex[0] = max([min([avgIndex[0], width]), 0])
    avgIndex[1] = max([min([avgIndex[1], height]), 0])

    return numpy.array([avgIndex])

def findSafeties(ten):
    #this only searches the first channel
    indices = (ten[0] != 0).nonzero().tolist()

    return len(indices)


#rotates an image and brute forces a rotation that keeps the image fit within bounds and happy :)
#work in progress
class SafeRotate(object):
    def __init__(self, prob):
        self.starting_pos = [0, 90, 180, 270]
        self.crop = SafeCropRescale(1, autoScale=False, crop_width = 326)
        self.minwidth=MIN_WIDTH
        self.minheight=MIN_HEIGHT
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

    def checkSafety(self, image, angle, safeties, buf=30):

        box = self.cropp_rotated(image, angle)

        if box[2] <= self.minheight or box[3] <= self.minwidth:
            return False

        rot_img = torchvision.transforms.functional.rotate(image, angle, expand=True, resample=False)
        count = 0

        for i in range(int(box[0]), int(box[0]+box[2])):
            for j in range(int(box[1]), int(box[1]+box[3])):
                if rot_img.getpixel((j, i))[0] > 0:
                    count += 1

        return count >= safeties - buf



    def __call__(self, x, y, label, safeties):

        if safeties == -1 or not happened(self.prob):
            return x, y, label, safeties

        start_pos = self.starting_pos[random.randint(0, 3)]

        x = torchvision.transforms.functional.rotate(x, start_pos, expand=True)
        y = torchvision.transforms.functional.rotate(y, start_pos, expand=True)

        label = calculateLabel(y)#findNewLabel(torchvision.transforms.functional.to_tensor(y))

        if start_pos == 90 or start_pos == 270:
            x, y, label, safeties = self.crop(x, y, label, safeties)

        possible_angles = [0]
        for i in range(5, 45, 5):
            theta = i
            if self.checkSafety(y, theta, safeties):
                possible_angles.append(theta)
            else:
                break

        for i in range(-5, -45, -5):
            theta = -1* i  #adjust for negative numbers
            if self.checkSafety(y, theta, safeties):
                possible_angles.append(theta)
            else:
                break

        final_angle = possible_angles[random.randint(0, len(possible_angles)-1)]

        #perform the final operations
        box = self.cropp_rotated(x, final_angle)

        x = torchvision.transforms.functional.rotate(x, final_angle, expand=True, resample=False)
        y = torchvision.transforms.functional.rotate(y, final_angle, expand=True, resample=False)
        x = torchvision.transforms.functional.resized_crop(x, box[0], box[1], box[2], box[3], (IMG_HEIGHT, IMG_WIDTH))
        y = torchvision.transforms.functional.resized_crop(y, box[0], box[1], box[2], box[3], (IMG_HEIGHT, IMG_WIDTH))

        label = calculateLabel(y)

        #image here is actually small (should be passed into safe crop and rescale for more processing)
        return x, y, label, -1


#chooses a random crop of the image within specified bounds (achieves a "translation" / crop / rescale)
class SafeCropRescale(object):

    def __init__(self, prob, autoScale=True, crop_width=None):
        self.ratio = IMG_HEIGHT/IMG_WIDTH #H/W
        self.minwidth=MIN_WIDTH
        self.minheight=MIN_HEIGHT
        self.height = IMG_HEIGHT
        self.width = IMG_WIDTH
        self.autoScale = autoScale
        self.crop_width = crop_width
        self.prob = prob

    def computeCropAndRescale(self, x, y, label):
        img_width, img_height = x.size

        #if image is too small, simply rescale
        if(img_width <= self.minwidth or img_height <= self.minheight):
            return x, y

        #if image is large enough continue with crop/rescale
        xc, yc = int(label[0, 0]*img_width), int(label[0,1]*img_height)

        width = self.crop_width if self.crop_width != None else random.randint(self.minwidth, img_width)
        height = self.ratio*width


        mbuffer = 30

        #compute coordinates of random top right corner for our width and height
        xlb = max(xc+mbuffer-width, 0)
        xub = min(xc-mbuffer, img_width-width)
        ylb = max(yc+mbuffer-height, 0)
        yub = min(yc-mbuffer, img_height-height)


        xfin = xub if xub <= xlb else random.randint(xlb, xub)
        yfin = yub if yub <= ylb else random.randint(int(ylb), int(yub))


        xn = torchvision.transforms.functional.crop(x, yfin, xfin, height, width)
        yn = torchvision.transforms.functional.crop(y, yfin, xfin, height, width)

        if self.autoScale:
            xn = torchvision.transforms.functional.resize(xn, (self.height, self.width))
            yn = torchvision.transforms.functional.resize(yn, (self.height, self.width))

        return xn, yn



    def __call__(self, x, y, label, safeties):

        if safeties == -1 or not happened(self.prob):
            return x, y, label, safeties

        x,y = self.computeCropAndRescale(x,y,label)
        tensor_label = torchvision.transforms.functional.to_tensor(y)
        label = calculateLabel(y)

        if self.autoScale:
            safeties = -1

        return x, y, label, safeties


#randomly flips the image and the label
class RandomFlip(object):
    def __init__(self, prob):
        self.prob = prob
        return

    def __call__(self, x, y, label, safeties):
        if happened(self.prob):
            x = func.hflip(x)
            y = func.hflip(y)
            val = -1*label[0,0].item()
            if val < 0:
                val = val + 1
            label[0,0] = val
        if(happened(self.prob)):
            x = func.vflip(x)
            y = func.vflip(y)
            val = -1*label[0,1].item()
            if val < 0:
                val = val + 1
            label[0,1] = val


        return x, y, label, safeties

