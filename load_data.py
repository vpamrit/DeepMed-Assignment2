import os
import random
import torch
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pandas import DataFrame, read_csv
from skimage import io, transform #might be the cause of torchvision failures


class SkinDataset(Dataset):

    def __init__(self, labels_file, root_dir, transform=None):
        self.labels = pd.read_csv(labels_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0];

    def get_df(self):
        return self.labels

    def get_label(self, idx):
        target_row = self.labels.iloc[idx, :]
        target = target_row.iloc[1:].to_numpy()

        label = np.where(target==1)[0]

        return label


    def get_data(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_row = self.labels.iloc[idx, :]
        img_name = target_row.iloc[0]
        image = io.imread(self.root_dir + img_name + ".jpg")

        if self.transform:
            image = self.transform(image)

        return img_name, image, self.get_label(idx)

    def __getitem__(self, idx):
        img_name, raw_image, label = self.get_data(idx)

        target = torch.from_numpy(label)
        image = torch.from_numpy(raw_image.transpose((2,0,1)))

        return image, target

class SampleSet:
    def __init__(self, num_classes=7):
        self.data = [set() for i in range(num_classes)]

    def __getitem__(self, idx):
        return self.data[idx]

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


class SkinDataLoader():
    def __init__(self, dataset, dist_emulation, batch_size, epoch_size=None, num_classes=7):
        self.data = dataset
        self.flatten = dist_emulation
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.avg = len(dataset)/num_classes
        self.num_classes = num_classes
        self.epoch = None

        if self.epoch_size == None:
            self.epoch_size = len(dataset)


        self.indices = SampleSet()

        #process df to extract the indices for each value [0: ...., 1: ...., 6: s3, s4, s5..]
        for i in range(len(dataset)):
            self.indices[dataset.get_label(i)[0]].add(i)

        #compute the desired number of each category in the epoch => populate the arrays
        self.num_samples = [int(len(samples) - (len(samples) - self.avg)*self.flatten) for samples in self.indices]
        total_sum = sum(self.num_samples)
        self.num_samples = [ int(float(x) / float(total_sum) * self.epoch_size) for x in self.num_samples ]

        total_sum = sum(self.num_samples)
        if total_sum < self.epoch_size:
            op = 1
        else:
            op = -1

        diff = abs(total_sum - self.epoch_size)

        while diff > 0:
            index = diff % 7
            if self.num_samples[index] > 0 or op == 1:
                self.num_samples[index] += op

            diff -= 1

        self.sampling_type = ["undersampling" if self.num_samples[i] < len(self.indices[i]) else "oversampling" for i in range(self.num_classes) ]

        #set up unused for future use
        self.unused = self.indices.clone()
        self.used = SampleSet()

        print(self.num_samples)
        print(sum(self.num_samples))

    def __getitem__(self, idx):
        return self.epoch[idx]

    def __len__(self):
        return len(self.epoch)

    def generate_epoch_samples(self):
        epoch_samples = []
        remaining_samples = self.num_samples.copy()

        for i in range(self.num_classes):
            category_samples = []
            if self.sampling_type[i] == "undersampling": ## if it is oversampling
                #attempt to put num_samples items in unused and fill self.used
                category_samples = SampleSet.pseudorandomly_undersample_to_list(self.unused[i], self.used[i], remaining_samples[i])
            elif self.sampling_type[i] == "oversampling": ## if it is undersampling
                category_samples = SampleSet.pseudorandomly_oversample_to_list(self.unused[i], remaining_samples[i])
            else:
                print("Unexpected value")
                quit()

            epoch_samples += category_samples

        self.epoch = epoch_samples
        return epoch_samples

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

    #def get_epoch_samples(self):
        # for each sample type: (from 1...7)
            # if undersampling:
                # init:
                # make a copy of the master list
                # regular:
                # first check if there are any unused (randomly sample from here)
                # add to current epoch samples
                # add it to used
                # once unused is empty and if you need more samples
                # swap used and unused
            # if oversampling:
                # put all items in the list n times (as many times as possible)
                # randomly sample the remainder to fit
                # put them in the sample list

        # populate arrays with appropriate amount from each category (add redundancies)
        # want to make sure to include unused samples (for the next iteration)
    #    return;


