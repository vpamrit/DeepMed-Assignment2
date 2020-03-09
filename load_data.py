import os
import PIL
import random
import torch
import torchvision
import numpy as np
import pandas as pd
import color_constancy as cc

from PIL import Image
from torch.utils.data import Dataset
from pandas import DataFrame, read_csv
from torchvision import transforms

import safetransforms


#target class will be a dict of targets
class SkinDataset(Dataset):

    def __init__(self, labels_file, root_dir, transform=False, target_classes=[0,1,2,3,4,5,6]):
        self.labels = pd.read_csv(labels_file)
        self.root_dir = root_dir
        self.transform = transform
        self.counter = 0
        self.target_classes = target_classes

    def set_targets(self, target_classes):
        self.target_classes = target_classes

    def __len__(self):
        return self.labels.shape[0];

    def get_df(self):
        return self.labels

    def get_label_by_img_name(self, img_name):
        target = self.labels.loc[self.labels['image'] == img_name]
        target = target.iloc[0].iloc[1:]
        label = np.where(target == 1)[0][0]

        return label

    def get_label(self, idx):
        target_row = self.labels.iloc[idx, :]
        target = target_row.iloc[1:].to_numpy()

        label = np.where(target==1)[0]

        #changes label type appropriately
        label = self.convert_label(label)

        return label

    def get_data(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_row = self.labels.iloc[idx, :]
        img_name = target_row.iloc[0]
        image = Image.open(self.root_dir + img_name + ".jpg")


        return img_name, image, self.get_label(idx)

    def __getitem__(self, idx):
        img_name, raw_image, label = self.get_data(idx)

        if self.transform:
            raw_image = self.exec_pil_transforms(raw_image)

        numpy_image = np.array(raw_image)
        numpy_image = cc.color_constancy(numpy_image)
        raw_image = Image.fromarray(numpy_image)
        raw_image = transforms.functional.resize((225, 300))
        #raw_image.save("./test-imgs/" + str(self.counter) + "img.jpg")

        #normalize and transform
        image = transforms.functional.to_tensor(raw_image)
        image = transforms.functional.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

        target = torch.from_numpy(label)

        return image, target

    def convert_label(self, label):

        #if default do nothing
        if not len(self.target_classes):
            return label

        #initialize to -1
        newlabel = -1*np.ones([1])

        #set the appropriate label to 1
        if label[0] in self.target_classes:
            newlabel[0] = self.target_classes.index(label[0])
        else:
            #represents "others" index
            if "others" in self.target_classes:
                newlabel[0] = len(self.target_classes) - 1

        return newlabel

    def exec_pil_transforms(self, pil_img):
        #consider a safe rotation here
        #or maybe a full rotation
        #pil_img.save("./test-imgs/" + str(self.counter) + "imgo.jpg")
        transform = torchvision.transforms.Compose([
            transforms.RandomHorizontalFlip(0.55),
            transforms.RandomVerticalFlip(0.55),
            transforms.ColorJitter(0.01, 0.01, 0.01, 0.01),
            transforms.RandomChoice(
                [
                    transforms.RandomResizedCrop((450, 600), scale=(0.7, 1.0)),
                    safetransforms.SafeRotate(0.85)
                ]
            ),
        ])

        pil_img = transform(pil_img)

        #pil_img.save("./test-imgs/" + str(self.counter) + "img.jpg")
        self.counter += 1

        return pil_img


class SampleSet:


    def __init__(self, num_classes=7):
        self.data = [set() for i in range(num_classes)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def size(self):
        total = 0

        for i in range(len(self.data)):
            total += len(self.data[i])

        return total

    def isEmpty(self):
        empty = 1
        for i in range(len(self.data)):
            empty = empty and bool(self.data(i))

        return empty

    def clone(self):
        return [pset.copy() for pset in self.data]

    def flatten(self):
        flat_list = []

        for i in range(len(self.data)):
            flat_list += list(self.data[i])

        return flat_list

    @staticmethod
    def pseudorandomly_undersample_to_list(origin, target_set, num_to_cpy):
        target_list = []

        num_origin_samples = len(origin)
        if num_origin_samples < num_to_cpy:
            target_list += list(origin).copy()
            target_set |= origin
            origin.clear()

            #sample in reverse
            return target_list + SampleSet.pseudorandomly_undersample_to_list(target_set, origin, num_to_cpy - num_origin_samples)

        return target_list + SampleSet.randomly_undersample_to_list(origin, target_set, num_to_cpy)

    @staticmethod
    def randomly_undersample_to_list(origin, target_set, num_to_cpy):
        target_list = []
        origin_list = list(origin)
        target_list += random.sample(origin_list, num_to_cpy)

        for i in range(len(target_list)):
            target_set.add(target_list[i])
            origin.remove(target_list[i])

        return target_list

    @staticmethod
    def pseudorandomly_oversample_to_list(origin, num_to_cpy):
        target_list = []

        iterations_to_copy = int(num_to_cpy / len(origin))
        remainder = num_to_cpy - iterations_to_copy*len(origin)

        for i in range(iterations_to_copy):
            target_list += list(origin)

        origin_list = list(origin)

        target_list += random.sample(origin_list, remainder)

        return target_list


#can behave as custom dataloader but single-threaded
class SkinDataManager(Dataset):
    def __init__(self, dataset, dist_emulation, batch_size, epoch_size=None, target_classes=[0,1,2,3,4,5,6], mode="soft"):
        self.data = dataset
        self.target_classes = target_classes
        self.data.set_targets(target_classes)

        self.flatten = dist_emulation
        self.num_classes = len(self.target_classes)
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.avg = len(dataset)/self.num_classes
        self.batch_num = 0
        self.epoch = None
        self.mode = mode

        self.unused = SampleSet(self.num_classes)

        #process df to extract the indices for each value [0: ...., 1: ...., 6: s3, s4, s5..]
        for i in range(len(dataset)):
            mclass = int(self.data.get_label(i)[0])

            #filter out unwanted indices
            if mclass != -1:
                self.unused[mclass].add(i)


        if self.epoch_size == None:
            self.epoch_size = self.unused.size()

        #compute the desired number of each category in the epoch => populate the arrays
        self.num_samples = [int(len(samples) - (len(samples) - self.avg)*self.flatten) for samples in self.unused]
        total_sum = sum(self.num_samples)
        self.num_samples = [ int(float(x) / float(total_sum) * self.epoch_size) for x in self.num_samples ]

        total_sum = sum(self.num_samples)
        if total_sum < self.epoch_size:
            op = 1
        else:
            op = -1

        diff = abs(total_sum - self.epoch_size)

        #fills the remaining slots available for the epoch
        while diff > 0:
            index = diff % self.num_classes
            if self.num_samples[index] > 0 or op == 1:
                self.num_samples[index] += op

            diff -= 1

        self.sampling_type = ["undersampling" if self.num_samples[i] < len(self.unused[i]) else "oversampling" for i in range(self.num_classes) ]

        #set up unused for future use
        self.used = SampleSet(self.num_classes)
        self.generate_epoch_samples()

        print(self.num_samples)
        print(sum(self.num_samples))

    def __getitem__(self, idx):
        return self.data[self.epoch[idx]]


    def __len__(self):
        return len(self.epoch)

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "soft":
            return next(super())

        if self.batch_size*self.batch_num >= len(self):
            self.generate_epoch_samples();
            raise StopIteration()
        else:
            return self.get_batch()

    def set_targets(self, target_classes):
        self.target_classes = target_classes
        self.data.set_targets(target_classes)

    def get_epoch_list(self):
        return self.epoch

    def generate_epoch_samples(self):
        self.batch_num = 0
        epoch_samples = []
        remaining_samples = self.num_samples

        for i in range(self.num_classes):
            category_samples = []
            if self.sampling_type[i] == "undersampling":
                #attempt to put num_samples items in unused and fill self.used
                category_samples = SampleSet.pseudorandomly_undersample_to_list(self.unused[i], self.used[i], remaining_samples[i])
            elif self.sampling_type[i] == "oversampling":
                category_samples = SampleSet.pseudorandomly_oversample_to_list(self.unused[i], remaining_samples[i])
            else:
                print("Unexpected value")
                quit()

            epoch_samples += category_samples

        self.epoch = epoch_samples
        random.shuffle(self.epoch)


    def get_batch(self):
        start_index = self.batch_num*self.batch_size
        newsample, newlabel = self.data[self.epoch[start_index]]
        samples, labels = newsample.unsqueeze(0), newlabel.unsqueeze(0)


        for i in range(start_index+1, min(start_index+self.batch_size, len(self))):
            newsample, newlabel = self.data[self.epoch[i]]
            samples = torch.cat((samples, newsample.unsqueeze(0)), 0)
            labels = torch.cat((labels, newlabel.unsqueeze(0)), 0)

        self.batch_num += 1
        return (samples, labels)

    def verify_bags(self, samples):
        print("VERIFYING BAG")
        print("SAMPLES: {}".format(len(samples)))

        calc_bags = [0, 0, 0, 0, 0, 0, 0]
        avg_bags = [0, 0, 0, 0, 0, 0, 0]

        for i in range(len(samples)):
            index = self.data.get_label(samples[i])[0]
            calc_bags[index] += 1;
            avg_bags[index] += samples[i]

        avg_bags = [float(avg_bags[i]) / float(calc_bags[i]) for i in range(len(avg_bags))]

        print("Total samples: {}".format(calc_bags))
        print("Average samples: ", end='')
        print(["{0:0.2f}".format(i) for i in avg_bags])

        return calc_bags == self.num_samples
