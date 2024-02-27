import pickle as pkl
import csv


annotations_file = '/home/adrienne/FGmovieAD/output/data_cache/MAD_val_annotations.pickle'
output_file = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/metadata/val.caption.linelist.tsv'

print('Loading cached annotations...')
annotations = pkl.load(open(annotations_file, 'rb'))
incr = 0

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations:
        #annotation_ID = annotation['id']

        writer.writerow([incr, 0])

        incr += 1