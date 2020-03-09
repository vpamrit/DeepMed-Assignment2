import pandas as pd
import argparse
import os

def main(args):
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
        epoch_size = row['epoch_size']

        save_folder = args.model_save_dir + model_type + "_" + str(i) + "/"

        #begin the train cycle here
        command = 'python3 train.py --{}_model {} \
                  --distribution_emulation_coefficient {} \
                  --model_save_dir {} \
                  --save_step 2 \
                  --epoch_size {} \
                  --target_classes {}'.format(model_name, model_num, balance, save_folder, epoch_size, target_classes)

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
