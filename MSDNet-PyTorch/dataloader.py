#This code is for Improving Los-Latency Predictions in Multi-Exit Architectures via Block-Dependent Losses
#Reference
#The original backbone code comes from
#https://github.com/kalviny/MSDNet-PyTorch


import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import pdb

def get_dataloaders(args):
    train_loader, val_loader, test_loader = None, None, None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        train_set = datasets.CIFAR10(args.data_root, train=True,
                                     transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                     ]))
        val_set = datasets.CIFAR10(args.data_root, train=False,
                                   transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                   ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(args.data_root, download=True, train=True,
                                      transform=transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize
                                      ]))
        val_set = datasets.CIFAR100(args.data_root, download=True, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
        if args.use_valid:
            train_set_index = torch.randperm(len(train_set))
            if os.path.exists(os.path.join(args.save, 'index.pth')):
                print('Load train_set_index')
                train_set_index = torch.load(os.path.join(args.save, 'index.pth'))
            else:
                print('Save train_set_index')
                torch.save(train_set_index, os.path.join(args.save, 'index.pth'))
            if args.data.startswith('cifar'):
                num_sample_valid = 5000
            else:
                num_sample_valid = 50000

            if 'train' in args.splits:
                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[:-num_sample_valid]),
                    num_workers=args.workers, pin_memory=True)
            if 'val' in args.splits:
                val_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=args.batch_size,
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                        train_set_index[-num_sample_valid:]),
                    num_workers=args.workers, pin_memory=True)
            if 'test' in args.splits:
                test_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
        else:
            if 'train' in args.splits:
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
            if 'val' or 'test' in args.splits:
                val_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                test_loader = val_loader
    else:
        # ImageNet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        val_set = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]))

        class_num, num_data_class = torch.unique(torch.Tensor(train_set.targets), return_counts=True)

        if args.use_valid:

            train_set_index = torch.randperm(len(train_set))
            if os.path.exists(os.path.join(args.save, 'index.pth')):
                print('Load train_set_index')
                train_set_index = torch.load('index.pth')
            else:
                print('Save train_set_index')
                torch.save(train_set_index, 'index.pth')
            if args.data.startswith('cifar'):
                num_sample_valid = 5000
            else:
                num_sample_valid = 50000

            if args.num_data == 1200:

                if 'train' in args.splits:
                    train_loader = torch.utils.data.DataLoader(
                        train_set, batch_size=args.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            train_set_index[:-num_sample_valid]),
                        num_workers=args.workers, pin_memory=True)
                if 'val' in args.splits:
                    val_loader = torch.utils.data.DataLoader(
                        train_set, batch_size=args.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            train_set_index[-num_sample_valid:]),
                        num_workers=args.workers, pin_memory=True)
                if 'test' in args.splits:
                    test_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

            else:



                if os.path.exists('num_data_' + str(args.num_data) +'index.pth'):
                    print('Load train sampling set_index')
                    dataset_idxs = torch.load('num_data_' + str(args.num_data) +'index.pth')
                else:
                    dataset_idxs = datasampling(train_set, train_set_index[-num_sample_valid:], args.num_data, 1000,
                                                num_data_class)
                    print('Save train sampling set_index')
                    torch.save(dataset_idxs, 'num_data_' + str(args.num_data) +'index.pth')




                if 'train' in args.splits:
                    train_loader = torch.utils.data.DataLoader(
                        DatasetSplit(train_set, dataset_idxs), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=True, shuffle=True)
                if 'val' in args.splits:
                    val_loader = torch.utils.data.DataLoader(
                        train_set, batch_size=args.batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            train_set_index[-num_sample_valid:]),
                        num_workers=args.workers, pin_memory=True)
                if 'test' in args.splits:
                    test_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)



        else:
            if args.num_data == 1200:
                if 'train' in args.splits:
                    train_loader = torch.utils.data.DataLoader(
                        train_set,
                        batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)
                if 'val' or 'test' in args.splits:
                    val_loader = torch.utils.data.DataLoader(
                        val_set,
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
                    test_loader = val_loader
            else:
                if os.path.exists('num_data_' + str(args.num_data) + 'index_novel.pth'):
                    print('Load train sampling set_index')
                    dataset_idxs = torch.load('num_data_' + str(args.num_data) + 'index_noval.pth')
                else:
                    dataset_idxs = datasampling(train_set, [], args.num_data, 1000,
                                                num_data_class)
                    print('Save train sampling set_index')
                    torch.save(dataset_idxs, 'num_data_' + str(args.num_data) + 'index_noval.pth')

                # pdb.set_trace()

            if 'train' in args.splits:
                train_loader = torch.utils.data.DataLoader(
                    DatasetSplit(train_set, dataset_idxs), batch_size=args.batch_size,
                    num_workers=args.workers, pin_memory=True, shuffle=True)
            if 'val' or 'test' in args.splits:
                val_loader = torch.utils.data.DataLoader(
                    val_set,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
                test_loader = val_loader

    return train_loader, val_loader, test_loader

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset= dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


import numpy as np

def datasampling(dataset, train_idx, num_data, num_classes, num_data_class):

    dataset_idxs = []
    idx_per_class = {}
    if len(train_idx) != 0:
        train_idx = train_idx.tolist()
    idxs = [i for i in range(len(dataset))]

    for j in range(num_classes):
        idx_per_class[j] = idxs[int(num_data_class[:j].sum()):int(num_data_class[:j+1].sum())]
        for i in idx_per_class[j]:
            if i in train_idx:
                idx_per_class[j].remove(i)




    for i in range(num_classes):
        data_idxs = np.random.choice(idx_per_class[i], num_data, replace=False)
        print(num_data)
        dataset_idxs += data_idxs.tolist()

    return dataset_idxs