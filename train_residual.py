import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import os
import model
import densenet as densemodel
import arlmodel
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

#custom imports
import load_data as ld
from load_data import SkinDataset, SkinDataManager
from model.residual_attention_network import ResidualAttentionModel_448input as ResidualAttentionModel

#global variables related to the image dataset properties
IMG_WIDTH = 600
IMG_HEIGHT = 450

def computeAccuracy(outputs, labels, num_classes):
    softm = torch.nn.functional.softmax(outputs.float(), dim=1)
    value, indices = torch.max(softm, dim=1)
    return (indices == labels.squeeze(1)).int().sum().item()

def main(args):
    #device configuration
    if args.cpu != None:
        device = torch.device('cpu')
    elif args.gpu != None:
        if not torch.cuda_is_available():
            print("GPU / cuda reported as unavailable to torch")
            exit(0)
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model directory
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    train_data = SkinDataset(labels_file=args.train_labels_file,
                           root_dir=args.train_image_dir, transform=True, binary_mode=(args.num_classes==2), target_class=args.target_class);

    validation_data = SkinDataset(labels_file=args.validation_labels_file,
                                   root_dir=args.validation_image_dir, binary_mode=(args.num_classes==2), target_class=args.target_class);

    train_manager = SkinDataManager(train_data, args.distribution_emulation_coefficient, args.batch_size, args.epoch_size)
    val_loader = DataLoader(dataset=validation_data, batch_size=args.validation_batch_size)


    # Build the models

    net = ResidualAttentionModel().to(device)

    if args.load_model != None:
        net.load_state_dict((torch.load(args.load_model)))

    params = net.parameters()

    # Loss and optimizer

    if args.loss_weights == None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(torch.FloatTensor(args.loss_weights).to(device))

    if args.optim != None:
        if args.optim == "adadelta":
            optimizer = torch.optim.Adadelta(params, lr=args.learning_rate)
        if args.optim == "adagrad":
            optimizer = torch.optim.Adagrad(params, lr=args.learning_rate)
        if args.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        if args.optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
        if args.optim == "rmsprop":
            optimizer = torch.optim.RMSProp(params, lr=args.learning_rate)
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(params, lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)


    val_loss_history = []
    train_loss_history = []
    failed_runs = 0
    prev_loss = float("inf")

    for epoch in range(args.num_epochs):
        train_loader = DataLoader(dataset=train_manager, batch_size=args.batch_size, shuffle=True);
        running_loss = 0.0
        total_loss = 0.0
        val_correct = 0
        train_correct = 0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            net.train()

            #adjust to output image coordinates
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs.float())

            #print(outputs)
            #print(labels)

            loss = criterion(outputs.float(), labels.squeeze().long())
            loss.backward()


            train_correct += computeAccuracy(outputs, labels, args.num_classes)

            if args.clipping_value != None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipping_value)

            optimizer.step()
            running_loss += loss.item()
            total_loss += loss.item()

            if i % 2 == 0: #print every mini-batches
              print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss/2))
              running_loss = 0.0

        loss = 0.0

        #compute validation loss at the end of the epoch
        for i, (inputs, labels) in enumerate(val_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            net.eval()
            with torch.no_grad():
                outputs = net(inputs.float())
                val_correct += computeAccuracy(outputs, labels, args.num_classes)
                loss += criterion(outputs, labels.squeeze().long()).item()

        print("------------------------------------------------------------")
        print("Epoch %5d" % (epoch+1))
        print("Training loss: {:.5f}, Avg Loss: {:.5f}".format(total_loss, total_loss / train_data.__len__()))
        print("Training Accuracy: {}".format(train_correct / len(train_manager)))
        print("Validation Loss: {:.5f}, Avg Loss: {:.5f}".format(loss, loss / validation_data.__len__()))
        print("Validation Accuracy: {}".format(val_correct/ len(validation_data)))
        print("------------------------------------------------------------")

        val_loss_history.append(loss)
        train_loss_history.append(total_loss)

        #save the model at the desired step
        if (epoch+1) % args.save_step == 0:
          torch.save(net.state_dict(), args.model_save_dir+"arlnet"+str(epoch+1)+".pt")

        ##stopping conditions
        if failed_runs > 5 and prev_loss < loss:
          break
        elif prev_loss < loss:
          failed_runs += 1
        else:
          failed_runs = 0

        prev_loss = loss
        train_manager.generate_epoch_samples()


    #create a plot of the loss
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1, len(val_loss_history)+1),val_loss_history,label="Validation loss")
    plt.plot(range(1,len(train_loss_history)+1), train_loss_history,label="Training loss")
    plt.xticks(np.arange(1, len(train_loss_history)+1, 1.0))
    plt.legend()
    plt.ylim((0, max([max(val_loss_history), max(train_loss_history)])))

    if args.save_training_plot != None:
        plt.savefig(args.save_training_plot+"loss_plot.png")

    plt.show()
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default='./data/train/', help='path to training set')
    parser.add_argument('--validation_image_dir', type=str, default='./data/validation/', help='path to validation set')
    parser.add_argument('--train_labels_file', type=str, default='./data/labels/Train_labels.csv', help='path to labels file for both validation and training datasets')
    parser.add_argument('--validation_labels_file', type=str, default='./data/labels/Validation_labels.csv', help='path to labels file for both validation and training datasets')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models/', help='path to location to save models')
    parser.add_argument('--save_step', type=int , default=4, help='step size for saving trained models')
    parser.add_argument('--num_classes', type=int , default=7, help='model to be trained as binary classifier')
    parser.add_argument('--target_class', type=int , default=0, help='model to be trained as binary classifier')
    parser.add_argument('--save_training_plot', nargs='?', type=str, const='./', help='location to save a plot showing testing and validation loss for the model')
    parser.add_argument('--load_model', type=str, default=None, help='Location of the saved model to load and then train')
    parser.add_argument('--pretrained', type=bool, default=False, help='Transfer learning (only works for densenet)')

    # Model parameters
    parser.add_argument('--distribution_emulation_coefficient', type=float, default=0.95, help="coefficient used to move the distribution of train data towards uniform (i.e. 0 is true distribution of train dataset, 1 is uniform distribution)")
    parser.add_argument('--loss_weights', type=float, nargs=7, help='input of weights where the sum is one - specifying how much the loss form each class should be weighted')
    parser.add_argument('--epoch_size', type=int, default=None, help="number of samples per epoch")
    parser.add_argument('--optim', type=str, default="adam", help="options such as adagrad, adadelta, sgd, etc.")
    parser.add_argument('--block_type', type=str, default="bottleneck", help='type of arlnet layer (bottleneck or basic)')
    parser.add_argument('--num_layers', type=int , nargs=4, help='input of four space-separated integers (i.e. 1 2 30 2 ) where each number represents the number of blocks at that respective layer')
    parser.add_argument('--arlnet_model', type=int , nargs=1, default=None, help='use to specify a pre-designed arlnet model (18 34 50 101 152). NOTE: this option will be overriden if both num_layers and block_type are specified')
    parser.add_argument('--densenet_model', type=int , nargs=1, default=169, help='use to specify a pre-designed densenet model (121 161 169 201)')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--validation_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--clipping_value', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--cpu', nargs='?', type=bool, const=True)
    parser.add_argument('--gpu', nargs='?', type=bool, const=True)
    args = parser.parse_args()

    print(args)
    main(args)