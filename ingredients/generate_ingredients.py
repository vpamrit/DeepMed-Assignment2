import csv

out_filename = './ingredients.csv'
categories = ['model_name',  'model_number', 'model_type', 'target_classes', 'balance', 'epoch_size']
rows = [
    ['resnet', 101, '1toMany', '2,others', 0.95, 2500],
    ['resnet', 50, '1to1', '1,2', 0.5, 2500],
    ['resnet', 50, 'All', '0,1,2,3,4,5,6', 0.90, 5000]
]

with open(out_filename, 'w', newline='') as outfile:
    writer=csv.writer(outfile, delimiter=',', skipinitialspace=True)
    writer.writerow(categories)

    for row in rows:
        writer.writerow(row)
