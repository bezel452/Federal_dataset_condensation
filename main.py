import os
import sys 
import torch 
import logging
from datetime import datetime
import json
import numpy as np
import random

from torch.utils.data import Subset
from networks import ConvNet
from config import parser
from datapre import partition, get_dataset, PerLabelDatasetNonIID
from client import FedNumClient
from server import FedNumServer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = parser.parse_args()
    args.dataset_root = args.dataset_root + '/' + args.dataset
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{current_time}_'FedNum'_{args.dataset}_alpha{args.alpha}_{args.client_num}clients_{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs_cr{args.communication_rounds}_dsa{args.dsa}.log"
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    class LoggerWriter:
        def __init__(self, logger, level):
            self.logger = logger
            self.level = level
            self.linebuf = ''

        def write(self, buf):
            for line in buf.rstrip().splitlines():
                self.logger.log(self.level, line.rstrip())

        def flush(self):
            pass
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    logging.info(f"Log file: {log_path}")
    logging.info(f"Arguments: {vars(args)}")

    split_file = f'/{args.dataset}_only_dirichlet_client_num={args.client_num}_alpha={args.alpha}.json'
    args.split_file = os.path.join(os.path.dirname(__file__), "split_file"+split_file)
    if not os.path.exists(args.split_file):
        logging.info(f"Split file not found: {args.split_file}")

        partition_args = type('Args', (), {
            'dataset': args.dataset,
            'client_num': args.client_num,
            'alpha': args.alpha,
            'dataset_root': args.dataset_root,
            'seed': args.seed
        })()
        partition(partition_args)
        logging.info(f"Data partition completed. File saved to: {args.split_file}")
    if args.dsa == 'True':
        args.dsa = True

    setup_seed(args.seed)
    device = torch.device(args.device)

    dataset_info, train_set, test_set, test_loader = get_dataset(args.dataset, args.dataset_root, args.batch_size)
    with open(args.split_file, 'r') as file:
        file_data = json.load(file)
    client_indices, client_classes = file_data['client_idx'], file_data['client_classes']

    logging.info("Checking data allocation...")
    empty_clients = []
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            empty_clients.append(i)
        logging.info(f"Client {i}: {len(indices)} samples, classes: {client_classes[i]}")
    
    if empty_clients:
        logging.warning(f"Found empty clients: {empty_clients}")
        logging.info("Filtering out empty clients to continue training...")
        
        # 过滤掉空客户端
        valid_clients = [i for i in range(args.client_num) if i not in empty_clients]
        client_indices = [client_indices[i] for i in valid_clients]
        client_classes = [client_classes[i] for i in valid_clients]
        
        # 更新客户端数量
        args.client_num = len(valid_clients)
        logging.info(f"Updated client_num to {args.client_num} (excluded {len(empty_clients)} empty clients)")

    train_sets = [Subset(train_set, indices) for indices in client_indices]
    
    '''
        数据集矩阵
    '''
    logging.info("The number of samples in each class...")
    for i, set_idx in enumerate(train_sets):
        client_class_list = [0 for _ in range(dataset_info['num_classes'])]
        for idx, (img, label) in enumerate(set_idx):
            client_class_list[label] += 1
        logging.info(f"Client {i}: samples in each class: {client_class_list}")


    if args.model == 'ConvNet':
        global_model = ConvNet(
            channel=dataset_info['channel'],
            num_classes=dataset_info['num_classes'],
            net_width=128,
            net_depth=3,
            net_act='relu',
            net_norm='instancenorm',
            net_pooling='avgpooling',
            im_size=dataset_info['im_size']
        ).to(device)
    else:
        raise NotImplementedError("Unknown Net.")
    
    model_identification = f'FedNum_{args.dataset}_alpha{args.alpha}_{args.client_num}clients/{args.model}_{args.ipc}ipc_{args.dc_iterations}dc_{args.model_epochs}epochs'

    client_list = [FedNumClient(
        cid = i,
        train_set = PerLabelDatasetNonIID(
            train_sets[i],
            client_classes[i],
            dataset_info['channel'],
            device,
        ),
        classes = client_classes[i],
        dataset_info = dataset_info,
        ipc = args.c_ipc,
        rho = args.rho,
        avg_num = args.avg_num,
        device = device,
        dsa_used = args.dsa,
        dc_iter = args.c_dc_iter,
        top_k = args.top_k
    ) for i in range(args.client_num)]

    server = FedNumServer(
        global_model = global_model,
        clients = client_list,
        communication_rounds = args.communication_rounds,
        join_ratio = args.join_ratio,
        batch_size = args.batch_size,
        model_epochs = args.model_epochs,
        ipc = args.ipc,
        rho = args.rho,
        avg_num = args.avg_num,
        batch_num = args.batch_num,
        dc_iterations = args.dc_iterations,
        image_lr = args.image_lr,
        eval_gap = args.eval_gap,
        test_set = test_set,
        test_loader = test_loader,
        device = device,
        model_identification = model_identification,
        dataset_info = dataset_info,
        dsa_used = args.dsa,
        dst_train = train_set,
        model_n = args.model_n,
        init_sample = args.init_sample
    )
    logging.info(f"Starting model training...")

    server.fit()
    server.final_eval()
    '''
    acc = server.evaluate()
    logging.info(f'Final evaluation: test acc is {acc}')
    '''
    logging.info(f"The model training completed.")

if __name__ == '__main__':
    main()