import copy
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision
from networks import ConvNet

import dsa

class FedNumServer:
    def __init__(
            self,
            global_model,
            clients,
            communication_rounds,
            join_ratio,
            batch_size,
            model_epochs,
            ipc,
            rho,
            avg_num,
            batch_num,
            dc_iterations,
            image_lr,
            eval_gap,
            test_set,
            test_loader,
            device,
            model_identification,
            dataset_info,
            dsa_used,
            dst_train,
            model_n,
            init_sample,
            init_img_save
    ):
        self.device = device
        self.global_model = global_model.to(device)
        self.clients = clients

        self.communication_rounds = communication_rounds
        self.join_ratio = join_ratio
        self.batch_size = batch_size
        self.model_epochs = model_epochs

        self.ipc = ipc
        self.rho = rho
        self.avg_num = avg_num
        self.batch_num = batch_num
        self.dc_iterations = dc_iterations
        self.image_lr = image_lr

        self.eval_gap = eval_gap
        self.test_set = test_set
        self.test_loader = test_loader

        self.model_identification = model_identification
        self.dataset_info = dataset_info

        all_classes = set()
        for clients in self.clients:
            all_classes.update(clients.classes)
        self.all_classes = sorted(list(all_classes))

        self.current_round = 0
        self.param = dsa.ParamDiffAug()

        self.dsa = dsa_used
        self.dst_train = dst_train
        self.init_img_save = init_img_save
        self.model_n = model_n
        if init_sample == 'real_sample':
            self.synthetic_images, self.synthetic_label = self.initialize_syn_data()
        elif init_sample == 'dm':
            self.synthetic_images, self.synthetic_label = self.initialize_dm()
        elif init_sample == 'random':
            self.synthetic_images, self.synthetic_label = self.initialize_rand()

    def initialize_syn_data(self):
        num_classes = len(self.all_classes)
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(self.dst_train[i][0], dim=0) for i in range(len(self.dst_train))]
        labels_all = [self.dst_train[i][1] for i in range(len(self.dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(self.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]
        
        image_syn = torch.randn(
            size=(num_classes * self.ipc, 
                  self.dataset_info['channel'],
                  self.dataset_info['im_size'][0],
                  self.dataset_info['im_size'][1]), 
                  dtype=torch.float, requires_grad=True, device=self.device
        )
        label_syn = torch.tensor([np.ones(self.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1)

        for c in range(num_classes):
            image_syn.data[c * self.ipc: (c + 1) * self.ipc] = get_images(c, self.ipc).detach().data

        return image_syn, label_syn
    
    def initialize_dm(self):
        num_classes = len(self.all_classes)
        synthetic_data = []
        synthetic_label = []
        selected_clients = self.select_clients()
        for client in selected_clients:
            img, labels, flag = client.initialize_dm()
            if flag == False:
                continue
            synthetic_data.append(img)
            synthetic_label.append(labels)
        indices_class = [[] for c in range(num_classes)]
        synthetic_data = torch.cat(synthetic_data, dim=0).to(self.device)
        synthetic_label = torch.cat(synthetic_label, dim=0).to(self.device)
        synthetic_label = torch.tensor(synthetic_label, dtype=torch.long, device=self.device)
        for i, lab in enumerate(synthetic_label):
            indices_class[lab].append(i)
        
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return synthetic_data[idx_shuffle]
        image_syn = torch.randn(
            size=(num_classes * self.ipc, 
                  self.dataset_info['channel'],
                  self.dataset_info['im_size'][0],
                  self.dataset_info['im_size'][1]), 
                  dtype=torch.float, requires_grad=True, device=self.device
        )
        label_syn = torch.tensor([np.ones(self.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1)
        
        for c in range(num_classes):
            image_syn.data[c * self.ipc: (c + 1) * self.ipc] = get_images(c, self.ipc).detach().data

        return image_syn, label_syn

    def initialize_rand(self):
        num_classes = len(self.all_classes)
        image_syn = torch.randn(
            size=(num_classes * self.ipc, 
                  self.dataset_info['channel'],
                  self.dataset_info['im_size'][0],
                  self.dataset_info['im_size'][1]), 
                  dtype=torch.float, requires_grad=True, device=self.device
        )
        label_syn = torch.tensor([np.ones(self.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=self.device).view(-1)

        return image_syn, label_syn

    def select_clients(self):
        return (
            self.clients if self.join_ratio == 1.0
            else random.sample(self.clients, int(round(len(self.clients) * self.join_ratio)))
        )

    def update_syn_data(self, client_data_list, seed, round):
        avg_features = []
        avg_logits = []
        avg_counts = []
        n_communication = 0
        for i in range(self.model_n):
            all_avg_features = {}
            all_avg_logits = {}
            all_avg_counts = {}

            for client_data in client_data_list:
                for c in client_data[i]['classes']:
                    if c not in all_avg_features.keys():
                        all_avg_features[c] = []
                        all_avg_logits[c] = []
                        all_avg_counts[c] = []
                    all_avg_features[c].append(client_data[i]['features'][c])
                    all_avg_logits[c].append(client_data[i]['logits'][c])
                    all_avg_counts[c].append(client_data[i]['counts'][c])
                    n_communication += len(client_data[i]['features'][c])
            
            for c in all_avg_features.keys():
        #     print(all_avg_features[c])
                all_avg_features[c] = torch.cat(all_avg_features[c], dim = 0)
                all_avg_logits[c] = torch.cat(all_avg_logits[c], dim = 0)
                all_avg_counts[c] = torch.cat(all_avg_counts[c], dim = 0)
            #    print(all_avg_features[c])
            avg_features.append(all_avg_features)
            avg_logits.append(all_avg_logits)
            avg_counts.append(all_avg_counts)

        print(f"Round {self.current_round}: communication volume: {n_communication}")

        optimizer_img = torch.optim.SGD([self.synthetic_images, ], lr = self.image_lr, momentum=0.5)
        optimizer_img.zero_grad()
        for dc_iteration in range(self.dc_iterations):
            
            indices_m = random.randint(0, self.model_n - 1)
            model_s = self.models[indices_m]
            all_avg_features = avg_features[indices_m]
            all_avg_logits = avg_logits[indices_m]
            all_avg_counts = avg_counts[indices_m]
    #        print(models)
            sample_model = copy.deepcopy(model_s)
            sample_model.train()
            for param in list(sample_model.parameters()):
                param.requires_grad = False

            loss = torch.tensor(0.0).to(self.device)
            
            for c in range(len(self.all_classes)):
                img_syn = self.synthetic_images[c * self.ipc: (c + 1) * self.ipc].reshape((self.ipc, 
                                                                                           self.dataset_info['channel'], 
                                                                                           self.dataset_info['im_size'][0],
                                                                                           self.dataset_info['im_size'][1]))
                torch.manual_seed(int(time.time() * 1000) % 100000)
                
                indices_div = torch.randperm(len(all_avg_counts[c]))
                target_avg_features = all_avg_features[c][indices_div]
                target_avg_logits = all_avg_logits[c][indices_div]
                target_avg_counts = all_avg_counts[c][indices_div]

                grs = self.batch_num // self.avg_num

                target_avg_counts = target_avg_counts[0:grs]    
                target_avg_features = target_avg_features[0:grs].to(self.device)
                target_avg_logits = target_avg_logits[0:grs].to(self.device)  
                '''
                target_avg_features = all_avg_features[c]
                target_avg_logits = all_avg_logits[c]
                target_avg_counts = all_avg_counts[c]
                '''
                
                if self.dsa:
            #        print(self.current_round)
                    img_syn = dsa.DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed[c], self.param)

                syn_features = sample_model.embed(img_syn)

                loss += torch.sum((torch.mean(target_avg_features, dim=0) - torch.mean(syn_features, dim=0)) ** 2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            if dc_iteration % 100 == 0:
                    print(f'Synthetic data update iteration {dc_iteration}, total loss = {loss.item() / len(self.all_classes)}')

    def fit(self):
        for round in range(self.communication_rounds):
            self.current_round = round

            if round == 0:
                if self.init_img_save == "True":
                    self.save_image("images_init/" + self.dataset_info['dataset'])
                    print(f"Save the initialized synthetic images in images_init.")
                
                print("Test initializing data...")
                
                syn_images, syn_labels = copy.deepcopy(self.synthetic_images.detach()), copy.deepcopy(self.synthetic_label.detach())

                synthetic_dataset = TensorDataset(syn_images, syn_labels)
                synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=0)
                
                self.global_model = ConvNet(
                        channel=self.dataset_info['channel'],
                        num_classes=self.dataset_info['num_classes'],
                        net_width=128,
                        net_depth=3,
                        net_act='relu',
                        net_norm='instancenorm',
                        net_pooling='avgpooling',
                        im_size=self.dataset_info['im_size']
                    ).to(self.device)
                
                self.global_model.train()  # 切换到训练模式
                lr = 0.01
                model_optimizer = torch.optim.SGD(
                    self.global_model.parameters(),
                    lr=lr,
                    weight_decay=0.0005,
                    momentum=0.9,
                )
                loss_function = torch.nn.CrossEntropyLoss().to(self.device)
                total_loss = 0
                lr_schedule = [self.model_epochs//2+1]
                for epoch in range(self.model_epochs + 1):
                    for x, target in synthetic_dataloader:
                        self.global_model.train()
                        # 数据移至设备
                        x, target = x.to(self.device), target.to(self.device)
                        target = target.long()
                        if self.dsa:
                            x = dsa.DiffAugment(x, 'color_crop_cutout_flip_scale_rotate', param=self.param)
                        # 前向传播
                        pred = self.global_model(x)
                        loss = loss_function(pred, target)

                        # 反向传播和参数更新
                        model_optimizer.zero_grad()
                        loss.backward()
                        model_optimizer.step()
                        total_loss += loss.item()
                        if epoch in lr_schedule:
                            lr *= 0.1
                            model_optimizer = torch.optim.SGD(
                                self.global_model.parameters(),
                                lr=lr,
                                weight_decay=0.0005,
                                momentum=0.9,
                            )
                acc = self.evaluate()
                print(f'[Init Data Test] epoch avg loss = {total_loss / self.model_epochs}, test acc = {acc}.')

            start_time = time.time()
            model_list = []
            for i in range(self.model_n):
                torch.manual_seed(int(time.time() * 1000) % 100000 + i)
                
                
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
                model_list.append(rnd_model)


            print('---------- client training ----------')
            self.models = model_list
#            self.models = self.global_model                                                                                                                             
            selected_clients = self.select_clients()

            seed = []
            for i in range(len(self.all_classes)):  # 这里可以改成每个模型的每个类都有一个对应的seed
                seed.append((int(time.time() * 1000 + i) % 100000 + i) % 10000000)
            client_data_list = []
            for client in selected_clients:
                client.receive_model(self.models)
                client_data = client.train(seed)
                client_data_list.append(client_data)
            self.update_syn_data(client_data_list, seed, round)

            if round % self.eval_gap == 0:
                '''
                    global training
                '''
                syn_images, syn_labels = copy.deepcopy(self.synthetic_images.detach()), copy.deepcopy(self.synthetic_label.detach())

                synthetic_dataset = TensorDataset(syn_images, syn_labels)
                synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=0)
                
                self.global_model = ConvNet(
                        channel=self.dataset_info['channel'],
                        num_classes=self.dataset_info['num_classes'],
                        net_width=128,
                        net_depth=3,
                        net_act='relu',
                        net_norm='instancenorm',
                        net_pooling='avgpooling',
                        im_size=self.dataset_info['im_size']
                    ).to(self.device)
                
                self.global_model.train()  # 切换到训练模式
                lr = 0.01
                model_optimizer = torch.optim.SGD(
                    self.global_model.parameters(),
                    lr=lr,
                    weight_decay=0.0005,
                    momentum=0.9,
                )
                loss_function = torch.nn.CrossEntropyLoss().to(self.device)
                total_loss = 0
                lr_schedule = [self.model_epochs//2+1]
                for epoch in range(self.model_epochs + 1):
                    for x, target in synthetic_dataloader:
                        self.global_model.train()
                        # 数据移至设备
                        x, target = x.to(self.device), target.to(self.device)
                        target = target.long()
                        if self.dsa:
                            x = dsa.DiffAugment(x, 'color_crop_cutout_flip_scale_rotate', param=self.param)
                        # 前向传播
                        pred = self.global_model(x)
                        loss = loss_function(pred, target)

                        # 反向传播和参数更新
                        model_optimizer.zero_grad()
                        loss.backward()
                        model_optimizer.step()
                        total_loss += loss.item()
                        if epoch in lr_schedule:
                            lr *= 0.1
                            model_optimizer = torch.optim.SGD(
                                self.global_model.parameters(),
                                lr=lr,
                                weight_decay=0.0005,
                                momentum=0.9,
                            )
                acc = self.evaluate()
                round_time = time.time() - start_time
                print(f'epoch avg loss = {total_loss / self.model_epochs}, test acc = {acc}, total time = {round_time}')

    def final_eval(self): 
        syn_images, syn_labels = copy.deepcopy(self.synthetic_images.detach()), copy.deepcopy(self.synthetic_label.detach())

        synthetic_dataset = TensorDataset(syn_images, syn_labels)
        synthetic_dataloader = DataLoader(synthetic_dataset, self.batch_size, shuffle=True, num_workers=0)
        
        self.global_model = ConvNet(
                channel=self.dataset_info['channel'],
                num_classes=self.dataset_info['num_classes'],
                net_width=128,
                net_depth=3,
                net_act='relu',
                net_norm='instancenorm',
                net_pooling='avgpooling',
                im_size=self.dataset_info['im_size']
            ).to(self.device)
        
        self.global_model.train()  # 切换到训练模式
        lr = 0.01
        model_optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=lr,
            weight_decay=0.0005,
            momentum=0.9,
        )
        loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        total_loss = 0
        model_epochs = 1000
        lr_schedule = [model_epochs//2+1]
        for epoch in range(model_epochs + 1):
            for x, target in synthetic_dataloader:
                self.global_model.train()
                # 数据移至设备
                x, target = x.to(self.device), target.to(self.device)
                target = target.long()
                if self.dsa:
                    x = dsa.DiffAugment(x, 'color_crop_cutout_flip_scale_rotate', param=self.param)
                # 前向传播
                pred = self.global_model(x)
                loss = loss_function(pred, target)

                # 反向传播和参数更新
                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()
                total_loss += loss.item()
                if epoch in lr_schedule:
                    lr *= 0.1
                    model_optimizer = torch.optim.SGD(
                        self.global_model.parameters(),
                        lr=lr,
                        weight_decay=0.0005,
                        momentum=0.9,
                    )
        acc = self.evaluate()
        print(f'FINAL EVAL------------epoch avg loss = {total_loss / model_epochs}, test acc = {acc}---------------')

    def evaluate(self):
        self.global_model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, target in self.test_loader:
                x, target = x.to(self.device), target.to(self.device)

                pred = self.global_model(x)
                _, pred_label = torch.max(pred.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

        return correct / float(total)
    
    def save_image(self, file):
        if not os.path.exists(file):
            os.mkdir(file)
        syn_images = copy.deepcopy(self.synthetic_images.detach())
        for i, img in enumerate(syn_images):
            file_path = file + '/' + str(i) + '.png'
            torchvision.utils.save_image(img, file_path)