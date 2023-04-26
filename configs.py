import jsonargparse

def get_config_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to the dataset', type=str, default='/notebooks/taxonomy.csv')
    parser.add_argument('--batch_size', help='batch sized used in train/val/test', type=int, default=16)
    parser.add_argument('--num_workers', help='number of workers for dataloader', type=int, default=4)
    parser.add_argument('--negative_ratio', help='percentage of negative pairs to sample compared to positve pairs', type=float, default=1)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)
    parser.add_argument('--seed', type=int, help='seed for reproducable training', default=123435)
    parser.add_argument('--precision', type=int, help='precision of floating point computation, normally this should not be changed',  default=32)
    parser.add_argument('--check_val_every_n_epoch', type=int, help='number of epoches between validation', default=5)
    parser.add_argument('--max_epochs', type=int, help='maximum number of epochs for training', default=10)
    parser.add_argument('--dev', help='whether to perform fast dev run for debugging purpose', action='store_true')
    parser.add_argument('--resume_from_checkpoint', type=str, help='checkpoint to resume traning from', default=None)
    parser.add_argument('--run_dir', type=str, help='directory to store files related to the training', default='/notebooks/runs')
    parser.add_argument('--model_name', type=str, help='name of the model file when saving', default='model')
    parser.add_argument('--project_name', type=str, help='project name for wandb logging', default='taxonomy')
    parser.add_argument('--weight_decay', type=float, help='weight decay (L2 regularization)', default=0.02)
    parser.add_argument('--prompt_learning', type=bool, help='whether to use prompt learning', default=False)
    parser.add_argument('--learnable_layers', type=int, help='number of transformer layers to unfreeze', default=0)


    return parser

def get_config(config_path):
    parser = get_config_parser()
    args = parser.parse_path(config_path)
    return args