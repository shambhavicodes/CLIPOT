from PIL import Image
import numpy as np
import os
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

def _add_channels(img):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while img.shape[-1] < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img

class TinyImageNetPaths:
    def __init__(self, root_dir, corruption='original', level=5):
        # Set up paths for the corrupted dataset
        val_path = os.path.join(root_dir, corruption, str(level))

        # Reference the original Tiny ImageNet paths
        root_dir_original = root_dir.replace('Tiny-ImageNet-C', 'tiny-imagenet-200')
        wnids_path = os.path.join(root_dir_original, 'wnids.txt')
        words_path = os.path.join(root_dir_original, 'words.txt')

        # Create paths and load class information
        self._make_paths(val_path, wnids_path, words_path)

    def _make_paths(self, corrupt_path, wnids_path, words_path):
        # Load the class IDs (WordNet IDs)
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)

        # Load the class names associated with each WordNet ID
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        # Set up paths for the corrupted dataset images
        self.paths = {'corrupt': []}  # [img_path, id, nid, box]
        corrupt_nids = os.listdir(corrupt_path)
        for nid in corrupt_nids:
            label_id = self.ids.index(nid)
            path = os.path.join(corrupt_path, nid)
            corrupt_name = os.listdir(path)
            for imgname in corrupt_name:
                fname = os.path.join(path, imgname)
                self.paths['corrupt'].append((fname, label_id, nid))


class TinyImageNetCDataset(Dataset):
    def __init__(self, root_dir, mode='corrupt', transform=None, max_samples=None, corruption='snow', level=5):
        # Initialize paths and labels using TinyImageNetPaths
        tinp = TinyImageNetPaths(root_dir, corruption=corruption, level=level)
        self.mode = mode
        self.label_idx = 1  # Index for label in the tuples [image_path, label_id, nid, box]
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)
        self.max_samples = max_samples

        # Load samples and class information
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        # Store class names for easy lookup
        self.classnames = {idx: tinp.nid_to_words[nid] for idx, nid in enumerate(tinp.ids)}

        # Limit the number of samples if max_samples is specified
        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s[0])
        img_array = np.array(img)
        if img_array.shape[-1] < 3 or len(img_array.shape) < 3:
            img_array = _add_channels(img_array)
            img = Image.fromarray(img_array)
        lbl = None if self.mode == 'test' else s[self.label_idx]

        if self.transform:
            sample = self.transform(img)
        return sample, lbl