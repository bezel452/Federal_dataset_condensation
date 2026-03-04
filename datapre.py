import torch
import numpy as np
from torchvision import datasets, transforms
import os
import json

def get_dataset(dataset, dataset_root, batch_size):
    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.CIFAR10(dataset_root, train=True, download=True, transform=transform)  # no augmentation
        testset = datasets.CIFAR10(dataset_root, train=False, download=True, transform=transform)
        class_names = trainset.classes
    else:
        raise NotImplementedError('Unknown Dataset.')
    
    dataset_info = {
        'channel': channel,
        'im_size': im_size,
        'num_classes': num_classes,
        'classes_names': class_names,
        'mean': mean,
        'std': std,
    }
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                  num_workers=2)
    return dataset_info, trainset, testset, testloader

def partition(args):
    np.random.seed(args.seed)

    if args.dataset == 'CIFAR10':
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dataset = datasets.CIFAR10(args.dataset_root, train=True, download=True, transform=transform)  # no augmentation
        class_names = dataset.classes
    else:
        raise NotImplementedError('Unknown Dataset.')
    
    K = num_classes
    labels = np.array(dataset.targets, dtype='int64')
    N = labels.shape[0]
    dict_users = {}

    for client_id in range(args.client_num):
        dict_users[client_id] = []
    
    for class_id in range(num_classes):
        idx_k = np.where(labels == class_id)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(args.alpha, args.client_num))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        idx_splits = np.split(idx_k, proportions)
        for i, client_id in enumerate(range(args.client_num)):
            dict_users[client_id].extend(idx_splits[i].tolist())

    for client_id in range(args.client_num):
        np.random.shuffle(dict_users[client_id])

    net_cls_counts = {}
    dict_classes = {}

    for net_i, dataidx in dict_users.items():
        unq, unq_cnt = np.unique(labels[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
        
        # 记录每个客户端拥有的类别（数据量大于0的类别）
        dict_classes[net_i] = [int(cls) for cls in unq]

    print('Data statistics: %s' % str(net_cls_counts))
    print('Client class assignments:')
    for client_id, classes in dict_classes.items():
        print(f'Client {client_id}: classes {classes} (total samples: {len(dict_users[client_id])})')

    save_path = os.path.join(os.path.dirname(__file__), 'split_file')
    file_name = f'{args.dataset}_only_dirichlet_client_num={args.client_num}_alpha={args.alpha}.json'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, file_name), 'w') as json_file:
        print(save_path)
        json.dump({
            "client_idx": [[int(idx) for idx in dict_users[i]] for i in range(args.client_num)],
            "client_classes": [[int(cls) for cls in dict_classes[i]] for i in range(args.client_num)],
        }, json_file, indent=4)

class PerLabelDatasetNonIID():
    def __init__(self, dst_train, classes, channel, device):  # images: n x c x h x w tensor
        self.images_all = []
        labels_all = []
        self.indices_class = {c: [] for c in classes}

        self.images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            if lab not in classes:
                continue
            self.indices_class[lab].append(i)
        self.images_all = torch.cat(self.images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    def __len__(self):
        return self.images_all.shape[0]

    def get_random_images(self, n):  # get n random images
        idx_shuffle = np.random.permutation(range(self.images_all.shape[0]))[:n]
        return self.images_all[idx_shuffle]

    def get_images(self, c, n, avg=False):  # get n random images from class c
        if not avg:
            if len(self.indices_class[c]) == 0:
                # If no samples for this class, return zero tensor with correct shape
                return torch.zeros(n, *self.images_all.shape[1:], device=self.images_all.device)
            elif len(self.indices_class[c]) >= n:
                idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
            else:
                sampled_idx = np.random.choice(self.indices_class[c], n - len(self.indices_class[c]), replace=True)
                idx_shuffle = np.concatenate((self.indices_class[c], sampled_idx), axis=None)
            return self.images_all[idx_shuffle]
        else:
            if len(self.indices_class[c]) == 0:
                # If no samples for this class, return zero tensor with correct shape
                return torch.zeros(n, *self.images_all.shape[1:], device=self.images_all.device)
            
            sampled_imgs = []
            for _ in range(n):
                if len(self.indices_class[c]) >= 4:
                    idx = np.random.choice(self.indices_class[c], 4, replace=False)
                else:
                    idx = np.random.choice(self.indices_class[c], 4, replace=True)
                sampled_imgs.append(torch.mean(self.images_all[idx], dim=0, keepdim=True))
            sampled_imgs = torch.cat(sampled_imgs, dim=0).cuda()
            return sampled_imgs