import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from data import train_test_split, get_loader
from utils import Capturing



class Network(nn.Module):

    def __init__(self, n_bands, n_classes, n_planes=2, patch_size=5):
        super(Network, self).__init__()
        self.n_bands = n_bands
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3))
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes, (3, 3, 3))
        self.features_size = self.fc_size()
        self.fc = nn.Linear(self.features_size, n_classes)
    
        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
    
    def fc_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.n_bands, self.patch_size, self.patch_size)
            )
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h


class CNN3D():

    def __init__(self, **kwargs):
        self.params = kwargs
        self.n_classes = self.params["n_classes"]
        self.n_bands = self.params["n_bands"]

        self.verbose = self.params.setdefault("verbose", True)
        self.device = self.params.setdefault("device", torch.device("cpu"))
        self.patch_size = self.params.setdefault("patch_size", 5)
        self.lr = self.params.setdefault("learning_rate", 0.01)
        self.epoch = self.params.setdefault("epoch", 250)
        self.batch_size = self.params.setdefault("batch_size", 100)

        weights = torch.ones(self.n_classes)
        weights[torch.LongTensor(self.params["ignored_labels"])] = 0.0
        weights = weights.to(self.device)
        self.weights = self.params.setdefault("weights", weights)


        self.net = Network(
            self.n_bands, self.n_classes, patch_size=self.patch_size
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=0.001
        )
        
        self.loss = nn.CrossEntropyLoss(
            weight=self.weights
        ).to(self.device)

        self.scheduler = self.params.setdefault(
            "scheduler",
            optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, threshold = 0.02, patience=self.epoch // 5, verbose=True
            ),
        )


    def fit(self, train_data):

        img, train_gt = train_data
        train_gt, val_gt = train_test_split(train_gt, 0.9)
        train_loader = get_loader(img, train_gt, shuffle=True, **self.params)
        val_loader = get_loader(img, val_gt, **self.params)

        self.train_accuracies = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []

        for e in tqdm(range(self.epoch), desc="Training the network"):
            
            # train step
            train_acc, train_loss = self.step(train_loader)
            self.train_accuracies.append(train_acc)
            self.train_losses.append(train_loss)
            
            # validation step
            val_acc, val_loss = self.step(val_loader, phase="Validation")
            self.val_accuracies.append(val_acc)
            self.val_losses.append(val_loss)

            # scheduler step
            with Capturing() as output:
                self.scheduler.step(val_loss)
            if len(output):
                tqdm.write(output[-1])

            # print output
            string =  "Epoch: {}/{}.. Training Loss: {:.4f}.. Validation Loss: {:.4f}.." +\
                " Training Accuracy: {:.4f}.. Validation Accuracy: {:.4f}"
            string = string.format(
                e+1, self.epoch, train_loss, val_loss, train_acc,val_acc
            )
            if self.verbose:
                tqdm.write(string)
            

    def step(self, data_loader, phase="Training"):
        if phase == "Training":
            self.net.train()
        else:
            self.net.eval()

        predicted_labels = []
        real_labels = []
        avg_loss = 0.0

        for data, target in tqdm(
            data_loader, total=len(data_loader), leave=False, desc=phase + " phase"
        ):
            data, target = data.to(self.device), target.to(self.device)
            if phase == "Training":
                self.optimizer.zero_grad()
                output = self.net(data.float()).to(self.device)

                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

            else:
                with torch.no_grad():
                    output = self.net(data)
                    loss = self.loss(output, target)

            avg_loss += loss.item()

            predicted_labels.append(output.argmax(dim=1))
            real_labels.append(target)

        avg_loss /= len(data_loader)

        predicted_labels = torch.cat(predicted_labels)
        real_labels = torch.cat(real_labels)

        acc = (predicted_labels == real_labels).type(torch.FloatTensor).mean()
        acc = float(acc)
        avg_loss=float(avg_loss)
        return acc, avg_loss


    def predict(self, data_loader):
        self.net.eval()
        predicted_labels = []
        real_labels = []
        for data, target in tqdm(
            data_loader, total=len(data_loader), leave=False
        ):
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                output = self.net(data)
                predicted_labels.append(output.argmax(dim=1))
                real_labels.append(target)

        predicted_labels = torch.cat(predicted_labels)
        real_labels = torch.cat(real_labels)
        return real_labels, predicted_labels
