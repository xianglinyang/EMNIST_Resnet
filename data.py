import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms
import torch

from torchvision.datasets.utils import makedir_exist_ok


class EMNISTData(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args

    def train_dataloader(self):
        # choose the first 10 letters
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)),
        ])
        dataset = EMNIST('data', split='digits', train=True, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)),
        ])
        dataset = EMNIST("data", split='digits', train=False, download=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def save_train_data(self, trainloader, path):
        trainset_data = None
        trainset_label = None
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if trainset_data != None:
                # print(input_list.shape, inputs.shape)
                trainset_data = torch.cat((trainset_data, inputs), 0)
                trainset_label = torch.cat((trainset_label, targets), 0)
            else:
                trainset_data = inputs
                trainset_label = targets

        training_path = os.path.join(path, "Training_data")
        makedir_exist_ok(training_path)
        # if not os.path.exists(training_path):
        #     os.mkdir(training_path)
        torch.save(trainset_data, os.path.join(training_path, "training_dataset_data.pth"))
        torch.save(trainset_label, os.path.join(training_path, "training_dataset_label.pth"))

    def save_test_data(self, testloader, path):

        testset_data = None
        testset_label = None
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if testset_data != None:
                # print(input_list.shape, inputs.shape)
                testset_data = torch.cat((testset_data, inputs), 0)
                testset_label = torch.cat((testset_label, targets), 0)
            else:
                testset_data = inputs
                testset_label = targets

        testing_path = os.path.join(path, "Testing_data")
        makedir_exist_ok(testing_path)
        # if not os.path.exists(testing_path):
        #     os.mkdir(testing_path)
        torch.save(testset_data, os.path.join(testing_path, "testing_dataset_data.pth"))
        torch.save(testset_label, os.path.join(testing_path, "testing_dataset_label.pth"))
