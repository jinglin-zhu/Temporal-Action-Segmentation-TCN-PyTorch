import os
import torch
from torch.utils.data import Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, bundle_file, features_dir, gt_dir, mapping_file):
        """
        Args:
            bundle_file: path to train.bundle and test.bundle
            features_dir: path to the directory containing the .npy feature files
            gt_dir: path to the directory containing the .txt ground truth files
            mapping_file: path to mapping.txt which contains lines of the format "class_id class_name"
        """
        # 1. read mapping file to create a dictionary {class_name: class_id}
        self.mapping = {}
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    self.mapping[parts[1]] = int(parts[0])

        # 2. read bundle file to get the list of video names (without extension)
        with open(bundle_file, 'r', encoding='utf-8') as f:
            self.video_list = [line.strip().split('.')[0] for line in f if line.strip()]

        self.features_dir = features_dir
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Args:
            idx: index of the video in the dataset
        Returns:
            features: (2048, T) tensor
            gt_indices: (T,) tensor of ground truth class indices corresponding to each frame
            video_name: the name of the video
        """
        video_name = self.video_list[idx]
        
        # read features from .npy file (shape: (2048, T))
        feature_path = os.path.join(self.features_dir, video_name + '.npy')
        features = np.load(feature_path)
        
        # read ground truth labels from .txt file and convert to class indices
        gt_path = os.path.join(self.gt_dir, video_name + '.txt')
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_labels = [line.strip() for line in f if line.strip()]
        
        gt_indices = [self.mapping[label] for label in gt_labels]

        # transform to tensors
        features = torch.tensor(features, dtype=torch.float32)
        gt_indices = torch.tensor(gt_indices, dtype=torch.long)

        return features, gt_indices, video_name

def collate_fn(batch):
    """
    padding for features and targets in a batch
     - features: pad to (batch_size, 2048, max_len) with 0
     - targets: pad to (batch_size, max_len) with -100 (ignore_index for CrossEntropyLoss)
    
    Args:
        batch: list of (features, targets, video_name) tuples
    Returns:
        padded_features: (batch_size, 2048, max_len)
        padded_targets: (batch_size, max_len)
        mask: (batch_size, max_len) 1.0 for valid frames, 0.0 for padding frames
        video_names: list of video names in the batch
    """
    features, targets, video_names = zip(*batch)
    
    lengths = [f.shape[1] for f in features]
    max_len = max(lengths)
    
    # pad features: in dimension 1 (time) to max_len, pad value 0
    padded_features = torch.zeros(len(batch), 2048, max_len)
    # pad targets: pad value -100 (ignore_index for CrossEntropyLoss)
    padded_targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    # Mask: 1.0 for valid frames, 0.0 for padding frames
    mask = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, (f, t) in enumerate(zip(features, targets)):
        padded_features[i, :, :f.shape[1]] = f
        padded_targets[i, :t.shape[0]] = t
        mask[i, :t.shape[0]] = 1.0

    return padded_features, padded_targets, mask, video_names