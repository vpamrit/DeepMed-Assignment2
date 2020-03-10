import csv

out_filename = './ingredients.csv'
categories = ['model_name',  'model_number', 'model_type', 'target_classes', 'balance', 'epoch_size']
rows = [
    ['densenet', 201, 'Most', '0,2,3,4,5,6', 0.6, 1500],
    ['densenet', 161, '1vs1', '0,1', 0.99, 1200],
    ['densenet', 161, '1vs1', '1,4', 0.99, 1200],
    ['densenet', 161, '1vsMany', '1,others', 0.95, 2000],
    ['densenet', 161, '1vsMany', '4,others', 0.85, 1000],
    ['densenet', 161, '1vsMany', '0,others', 0.85, 1000],
    ['densenet', 201, 'All', '0,1,2,3,4,5,6', 0.8, 3500],
]

with open(out_filename, 'w', newline='') as outfile:
    writer=csv.writer(outfile, delimiter=',', skipinitialspace=True)
    writer.writerow(categories)

    for row in rows:
        writer.writerow(row)
