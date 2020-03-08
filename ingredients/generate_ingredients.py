import csv

out_filename = './ingredients.csv'
categories = ['model_name', 'model_type', 'categories', 'balanced']
rows = [
    ['resnet101', '1toMany', '2,others', 0.95],
    ['resnet52', '1to1', '1,2', 0.5],
    ['resnet52', 'All', '0,1,2,3,4,5,6', 0.90]
]

with open(out_filename, 'w', newline='') as outfile:
    writer=csv.writer(outfile, delimiter=',', skipinitialspace=True)
    writer.writerow(categories)

    for row in rows:
        writer.writerow(row)
