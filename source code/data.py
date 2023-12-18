import numpy as np
import torch
import torch.utils
import torch.utils.data
from scipy import io
import sklearn.model_selection as sk_ms


def get_dataset(img_path, gt_path):
    """
    Loads the dataset.

    Parameters
    ----------
    img_path: str, path to image file
    gt_path: str, path to ground truth data

    Returns
    ----------
    img: 3darray, spectral image of scene
    gt: 2darray, ground truth labels
    label_values: list, target names
    ignored_labels: list, labels to ignore(e.g. 'Undefined')
    valid_bands: list, 90% of initial bands with most amount of \
    samples within 3 sigma interval
    """

    
    img = io.loadmat(img_path)["paviaU"]
    gt = io.loadmat(gt_path)["paviaU_gt"]

    label_values = [
        "Undefined",
        "Asphalt",
        "Meadows",
        "Gravel",
        "Trees",
        "Painted metal sheets",
        "Bare Soil",
        "Bitumen",
        "Self-Blocking Bricks",
        "Shadows",
    ]

    ignored_labels = [0]

    # drop 10% of noisiest bands
    band_zscore = np.abs(img - np.mean(img, axis=(0,1))[None, None, :]) / np.std(img, axis=(0,1))[None, None, :]
    noised_ratio = (abs(band_zscore) >= 3).mean((0,1))
    valid_bands = noised_ratio.argsort()[:-img.shape[2]//10]

    # normalize data
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    return img, gt, label_values, ignored_labels, valid_bands


class DL(torch.utils.data.Dataset):
    """
    Custom torch Data Loader partciluarly for this task
    """

    def __init__(self, img, gt, patch_size, ignored_labels):
        """
        Parameters
        ----------
        img: 3darray, spectral image of scene
        gt: 2darray, ground truth labels
        patch_size: int, speaks for itself
        ignored_labels: list, labels to ignore(e.g. 'Undefined')
        """
        super(DL, self).__init__()
        self.data = img
        self.labels = gt
        self.patch_size = patch_size
        self.ignored_labels = ignored_labels

        # mask of unignored labels
        mask = np.ones_like(gt)
        mask[np.isin(gt, self.ignored_labels)] = 0
        
        # select valid indices due to patching
        i_nz, j_nz = np.nonzero(mask)
        p = self.patch_size // 2
        
        indx_mask = np.logical_and.reduce((
            p < i_nz, i_nz < self.data.shape[0] - p,
            p < j_nz, j_nz < self.data.shape[1] - p
        ))
        self.indices = np.array(list(zip(i_nz[indx_mask], j_nz[indx_mask])))

        # shuffle to feed to network in random way
        np.random.shuffle(self.indices)

    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        # select sample
        x, y = self.indices[i]
        
        # select patch of sample
        x_l, y_l = x - self.patch_size // 2, y - self.patch_size // 2
        x_r, y_r = x_l + self.patch_size, y_l + self.patch_size

        data = self.data[x_l:x_r, y_l:y_r]
        labels = self.labels[x_l:x_r, y_l:y_r]

        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        labels = np.asarray(np.copy(labels), dtype="int64")
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)
        
        # getting label of center pixel of the patch
        label = labels[self.patch_size // 2, self.patch_size // 2]

        # add batch dim
        data = data.unsqueeze(0)

        return data, label



def train_test_split(gt, train_size):
    """
    Splits the data into train and test.

    Parameters
    ----------
    gt: 2darray, ground truth labels
    train size: int in range (0,1), proportion of data\
        to include in train split
    
    Returns
    ----------
    train_gt: 2darray, train split of ground truth
    test_gt: 2darray, test split of ground truth
    """
    indices = np.nonzero(gt)
    data = list(zip(*indices))
    labels = gt[indices].ravel()

    # fill with ignored labels
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)

    train_indices, test_indices = sk_ms.train_test_split(
        data, train_size=train_size, stratify=labels
    )
    
    train_indices = tuple(np.transpose(train_indices))
    test_indices = tuple(np.transpose(test_indices))
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]

    return train_gt, test_gt


def get_loader(img, gt, shuffle=False, **hyperparams):
    """
    Creates torch DataLoader

    Parameters
    ----------
    shuffle: bool, whether to shuffle the dataset
    **hyperparams: unpacked dict, should contain 'patch_size', \
    'batch_size' and 'ignored' labels (added in main by default)

    Returns
    ----------
    loader: torch.utils.data.DataLoader
    """
    dataset = DL(
        img, gt,
        hyperparams["patch_size"],
        hyperparams["ignored_labels"]
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=shuffle
    )
    return loader