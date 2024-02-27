import h5py
import torch
import pickle as pkl
import csv

def uniform_subsample(tensor, num_samples):
    """
    Uniformly subsample 'num_samples' frames from the tensor.
    """
    total_frames = tensor.size(0)
    if total_frames == 0 or total_frames < num_samples:
        return tensor
    indices = torch.linspace(0, total_frames - 1, num_samples).long()
    return tensor[indices]

video_features_file = '/mnt/welles/scratch/adrienne/MAD/features/CLIP_B32_frames_features_5fps.h5'
annotations_file = '/home/adrienne/FGmovieAD/output/data_cache/MAD_train_annotations.pickle'
output_file = '/home/adrienne/MAD-FAVD/AVLFormer/datasets/8_frames_tsv/train_segmented_features.tsv'

frames_sampling = 8

print('Loading cached annotations...')
annotations = pkl.load(open(annotations_file, 'rb'))
movies = {a['movie']:a['movie_duration'] for a in annotations}

print('Loading video features...')
with h5py.File(video_features_file, 'r') as f:
    video_feats = {m: torch.from_numpy(f[m][:]) for m in movies}

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter='\t')

    for annotation in annotations:
        annotation_ID = annotation['id']
        movie_ID = annotation['movie']
        start_frame, end_frame = annotation['frames_idx']

        print(f'Processing movie ID: {movie_ID}')
        full_feature_tensor = video_feats[movie_ID]

        segmented_feature_tensor = full_feature_tensor[start_frame:end_frame + 1]
        subsampled_feature_tensor = uniform_subsample(segmented_feature_tensor, frames_sampling)

        # Write annotation ID and frame tensors in one line
        row = [annotation_ID] + [frame_tensor.numpy().tolist() for frame_tensor in subsampled_feature_tensor]
        writer.writerow(row)





