import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=2147483647)
parser.add_argument("--device", type=str, default="cuda:1")

parser.add_argument("--dataset_root", type=str, default="/home/data/datasets")
parser.add_argument("--dataset", type=str, default='CIFAR10')

parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--dsa", type=bool, default=True,
                    help="使用dsa")

parser.add_argument("--model", type=str, default="ConvNet")
parser.add_argument("--communication_rounds", type=int, default=20)

parser.add_argument("--batch_num", type=int, default=256,
                    help='server上根据随机挑选多个同一类的真实样本的平均特征和平均logits来优化该类的合成数据')
parser.add_argument("--avg_num", type=int, default=32,
                    help='客户端每次将同一个类的avg_num个样本的特征和logits取平均后上传')

parser.add_argument("--ipc", type=int, default=50)
parser.add_argument("--rho", type=int, default=5)
parser.add_argument("--dc_iterations", type=int, default=1000)
parser.add_argument("--dc_batch_size", type=int, default=256)
parser.add_argument("--image_lr", type=float, default=1.0)

parser.add_argument("--join_ratio", type=float, default=1.0)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--model_epochs", type=int, default=1000)

parser.add_argument("--client_num", type=int, default=50)
parser.add_argument("--eval_gap", type=int, default=1)

parser.add_argument("--model_n", type=int, default=1, help="The number of models")

parser.add_argument("--init_sample", type=str, default="real_sample", help="the method of data initialization")
parser.add_argument("--c_ipc", type=int, default=50, help="client dm ipc")
parser.add_argument("--c_dc_iter", type=int, default=1000)

parser.add_argument("--top_k", type=int, default=3, help="select top k class in each client")