import copy
from typing import List
import time

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from datapre import PerLabelDatasetNonIID
import dsa

from networks import ConvNet

class FedNumClient:
    def __init__(
            self,
            cid: int,
            train_set: PerLabelDatasetNonIID,
            classes: List[int],
            dataset_info: dict,
            ipc: int,
            rho: float,
            avg_num: int,
            device: torch.device,
            dsa_used: bool,
            dc_iter: int,
            top_k: int
    ):
        self.cid = cid
        self.train_set = train_set
        self.classes = classes
        self.dataset_info = dataset_info
        self.ipc = ipc
        self.rho = rho
        self.avg_num = avg_num
        self.param = dsa.ParamDiffAug()
        self.device = device
        self.dsa = dsa_used
        self.dc_iter = dc_iter
        self.top_k = top_k
        self.images = self.initialize_data()
        '''
        if init_sample == 'dm':
            self.synthetic_images, self.synthetic_labels = self.initialize_dm()
        '''

    def shuffle_data(self, images):
        transform = transforms.Compose([
            # 随机水平翻转，50%概率
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机垂直翻转，20%概率
            transforms.RandomVerticalFlip(p=0.2),
            # 随机旋转，角度范围在-15到15度之间
            transforms.RandomRotation(degrees=(-15, 15), expand=False),
            # 随机调整亮度、对比度和饱和度
            transforms.ColorJitter(
                brightness=0.15,  # 亮度调整范围
                contrast=0.15,    # 对比度调整范围
                saturation=0.15   # 饱和度调整范围
            ),
            # 随机擦除，30%概率应用，模拟遮挡
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.15),  # 擦除区域面积占比
                ratio=(0.3, 3.3),    # 擦除区域宽高比
                value=0.5            # 擦除区域填充值（灰色）
            )
        ])
        # 对批次中的每个图像应用变换
        augmented_images = []
        for img in images:
            augmented_img = transform(img)
            augmented_images.append(augmented_img)
        
        # 堆叠成批次张量并返回
        return torch.stack(augmented_images)

    def initialize_data(self):
        images = {}
        for c in self.classes:
            available_samples = len(self.train_set.indices_class[c])
            if available_samples == 0:
                print(f"Client {self.cid} has no samples for class {c}, so skipping ... ")
                continue
            all_real_images = []
            indices = self.train_set.indices_class[c]
            for idx in indices:
                all_real_images.append(self.train_set.images_all[idx].unsqueeze(0))

            if len(all_real_images) == 0:
                images[c] = []
                continue

            original_images = torch.cat(all_real_images, dim=0)

            images_s = original_images.size(0)
            all_real_images_expanded = [original_images]

            grs = images_s // self.avg_num
            while True:
                torch.manual_seed(int(time.time() * 1000) % 100000)
                div = self.avg_num - (images_s - self.avg_num * grs)
                choice_s = min(div, images_s)
                shuffled_indices = torch.randint(0, images_s, (choice_s,))
                tmp_images = torch.cat(all_real_images_expanded, dim=0)
                shuffled_images = self.shuffle_data(tmp_images[shuffled_indices])
                all_real_images_expanded.append(shuffled_images)
                images_s += choice_s
                if images_s % self.avg_num == 0:
                    break

            all_real_images = torch.cat(all_real_images_expanded, dim=0)
            all_real_images = all_real_images.to(self.device)
            images[c] = all_real_images
        return images
    
    def initialize_dm(self):
        selected_classes = []
        imgs = []
        for c in self.classes:
            imgs.append(len(self.train_set.indices_class[c]))

        top_k_idx = sorted(imgs, reverse=True)[:min(self.top_k, len(self.classes))]
        
        for i, c in enumerate(self.classes):
            for idx in top_k_idx:
                if idx == len(self.train_set.indices_class[c]):
                    selected_classes.append(c)
                    break
        print(f"Client {self.cid}: Classes {selected_classes} are used to initialize synthetic data...")
        synthetic_images = torch.randn(
                size=(len(selected_classes) * self.ipc, 
                    self.dataset_info['channel'],
                    self.dataset_info['im_size'][0],
                    self.dataset_info['im_size'][1]), 
                    dtype=torch.float, requires_grad=True, device=self.device
                )

        for i, c in enumerate(selected_classes):
            synthetic_images.data[i * self.ipc : (i + 1) * self.ipc] = self.train_set.get_images(c, self.ipc, avg=False).detach().data
        optimizer_image = torch.optim.SGD([synthetic_images], lr=1, momentum=0.5, weight_decay=0)
        optimizer_image.zero_grad()

        for dc in range(self.dc_iter):
            torch.manual_seed(int(time.time() * 1000) % 100000)
            rnd_model = ConvNet(
                channel=self.dataset_info['channel'],
                num_classes=self.dataset_info['num_classes'],
                net_width=128,
                net_depth=3,
                net_act='relu',
                net_norm='instancenorm',
                net_pooling='avgpooling',
                im_size=self.dataset_info['im_size']
            ).to(self.device)
            rnd_model.eval()
            loss = torch.tensor(0.0).to(self.device)
        
            for i, c in enumerate(selected_classes):
                real_image = self.train_set.get_images(c, 256)
                synthetic_image = synthetic_images[i * self.ipc : (i + 1) * self.ipc].reshape(
                    (self.ipc, self.dataset_info['channel'], self.dataset_info['im_size'][0], self.dataset_info['im_size'][1])
                )
                real_image = real_image.to(self.device)

                real_feature = rnd_model.embed(real_image).detach()
                synthetic_feature = rnd_model.embed(synthetic_image)
                real_logits = rnd_model(real_image).detach()
                synthetic_logits = rnd_model(synthetic_image)

                loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0))**2)
                loss += torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0))**2)

            optimizer_image.zero_grad()
            loss.backward()
            optimizer_image.step()
            if dc % 100 == 0:
                print(f'[Initialization] client{self.cid}, DM {dc}, avg loss = {loss.item() / len(selected_classes)}')

        synthetic_labels = torch.cat([torch.ones(self.ipc) * c for c in selected_classes])
        return copy.deepcopy(synthetic_images.detach()), synthetic_labels
        
    def get_features_logits(self, model, seed):
        model.eval()

        features = {}
        logits = {}
        counts = {}

        with torch.no_grad():
            for c in self.classes:
                indices_div = torch.randperm(len(self.images[c]))
                all_real_images = self.images[c][indices_div]
                if len(all_real_images) == 0:
                    continue
                data_c = all_real_images
                if self.dsa:
                    data_c = dsa.DiffAugment(data_c, 'color_crop_cutout_flip_scale_rotate', seed[c], self.param)
                    data_c = data_c.to(self.device)

                all_features = model.embed(data_c).detach()
                all_logits = model(data_c)
 #               print(all_features.size())
                num_samples = all_features.size(0)
        
                num_groups = num_samples // self.avg_num

                avg_features_list = []
                avg_logits_list = []
                avg_counts_list = []

                for group_idx in range(num_groups):
                    start_idx = group_idx * self.avg_num
                    end_idx = min((group_idx + 1) * self.avg_num, num_samples)
                    
                    group_features = all_features[start_idx: end_idx]
                    group_logits = all_logits[start_idx: end_idx]

                    group_size = end_idx - start_idx

                    avg_feature = torch.mean(group_features, dim=0)
                    avg_logits = torch.mean(group_logits, dim=0)

                    avg_features_list.append(avg_feature)
                    avg_logits_list.append(avg_logits)
                    avg_counts_list.append(group_size)

                features[c] = torch.stack(avg_features_list)
                logits[c] = torch.stack(avg_logits_list)
                counts[c] = torch.tensor(avg_counts_list)

        return {
            "classes": self.classes,
            "features": features,
            "logits": logits,
            "counts": counts,
        }

    def train(self, seed):
        l = len(self.models)
        client_data = []
        for i in range(l):
            sample_model = copy.deepcopy(self.models[i])
            client_data.append(self.get_features_logits(sample_model, seed))
        return client_data


    def receive_model(self, models):
        self.models = models
