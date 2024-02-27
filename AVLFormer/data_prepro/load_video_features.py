import sys
import torch
import pickle as pkl
import csv
from itertools import islice

MAD_features = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/8_frames_tsv/train_segmented_features.tsv'
FAVD_features = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/frame_tsv/val_32frames.img.tsv'

csv.field_size_limit(sys.maxsize)

print('Loading MAD video features...')
with open(MAD_features, 'r', encoding='utf-8') as MAD_file:
    # Create a CSV reader object specifying the delimiter as a tab
    MAD_reader = csv.reader(MAD_file, delimiter='\t')

    MAD_data = [row for row in islice(MAD_reader, 10)]
    print('completed MAD load in')


print('Loading FAVD video features...')
with open(FAVD_features, 'r', encoding='utf-8') as FAVD_file:
    # Create a CSV reader object specifying the delimiter as a tab
    FAVD_reader = csv.reader(FAVD_file, delimiter='\t')

    FAVD_data = [row for row in islice(FAVD_reader, 10)]
    print('completed FAVD load in')
