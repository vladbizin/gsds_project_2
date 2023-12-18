from utils import get_device
from data import get_dataset, train_test_split, get_loader
from models import CNN3D
from utils import save_params
import torch

CUDA_DEVICE = get_device()

img, gt, LABEL_VALUES, IGNORED_LABELS, VALID_BANDS = get_dataset(
    "PaviaU.mat",
    "PaviaU_gt.mat"
    )
N_CLASSES = len(LABEL_VALUES)
N_BANDS = img.shape[-1]

hyperparams ={
    "ignored_labels": IGNORED_LABELS,
    "n_classes": N_CLASSES,
    "n_bands": N_BANDS,
    "device": CUDA_DEVICE,

    "verbose": True,
    "patch_size": 5,
    "learning_rate": 0.01,
    "epoch": 250,
    "batch_size": 100,
}

for i in range(5):
    train_gt, test_gt = train_test_split(gt, 0.5)
    model = CNN3D(**hyperparams)

    model.fit([img, train_gt])

    to_save = ({
        "train_gt": train_gt,
        "test_gt": test_gt,
        "train_acc": model.train_accuracies,
        "val_acc": model.val_accuracies,
        "train_loss": model.train_losses,
        "val_loss": model.val_losses,
        "params": model.params
    })

    # save_params(to_save, "no_dropout_" + str(i+1) +".pkl")
    # torch.save(model.net.state_dict(), "no_dropout_" + str(i+1) +".pt")