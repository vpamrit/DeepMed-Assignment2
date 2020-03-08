
# this class should be used for training that's fine
# going to pass in last layer architecture as well as what classes it should be predicting
# this should have a cascading effect on the data loader (what kind of data is being loaded in)
# store models appropriately (if training fails somewhere we can always pick up again)
# we want to preprocess normalize and rescale to 224 x 224 => we are going to be using only resnet101s
# testing architecture should read in csv and load models appropriately

import pandas as pd
import argparse
import os

def main(args):
    #construct a one-to-one (as dictated between classes)
    #construct a one-to-many (specify number of epochs)
    #we are going to change the way train.py works fundamentally to make this work and how the dataloader works as well

    models_df = pd.read_csv(args.models_csv)
    print(models_df)

    if not os.path.isdir(args.model_save_dir):
        os.mkdir(args.model_save_dir)

    for i, row in models_df.iterrows():
        model_name = row['model_name']
        model_num = row['model_number']
        model_type = row['model_type']
        target_classes = row['target_classes']
        balance = row['balance']

        save_folder = args.model_save_dir + model_type + "_" + str(i)

        #begin the train cycle here
        command = 'python3 train.py --{}_model {} \
                  --distribution_emulation_coefficient {} \
                  --model_save_dir {} \
                  --target_classes {}'.format(model_name, model_num, balance, save_folder, target_classes)

        print(command)

        os.system(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, default='./data/train/', help='path to training set')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models/', help='path to saved models')
    parser.add_argument('--models_csv', type=str, default='./ingredients/ingredients.csv', help='path to csv to create ensemble')
    args = parser.parse_args()
    print(args)
    main(args)
