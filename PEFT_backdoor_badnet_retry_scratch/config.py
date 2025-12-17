import argparse

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='mnist', help='Which dataset to use (mnist or cifar10, default: mnist)')
parser.add_argument('--no_train', action='store_false', help='train model or directly load model (default true, if you add this param, then without training process)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optim', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--trigger_label', type=int, default=0, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train backdoor model, default: 50')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--pp', action='store_true', help='Do you want to print performance of every label in every epoch (default false, if you add this param, then print)')
parser.add_argument('--datapath', default='./dataset/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--poisoned_portion', type=float, default=0.1, help='posioning portion (float, range from 0 to 1, default: 0.1)')

# newly added.
parser.add_argument('--lora_rank', type=int, default=16, help='Rank for LoRA')
parser.add_argument('--lora_alpha', type=float, default=16.0, help='Alpha scaling for LoRA')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout for LoRA')
parser.add_argument('--freeze_weights', action='store_true', help='Freeze base weights if set')
parser.add_argument('--use_lora', action='store_true', help='Use LoRA-based backdoor if set')


parser.add_argument('--checkpoint_name', default='badnet-cifar10.pth', help='Where to save the model')
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
parser.add_argument('--data_aug', action='store_true', help='Use data augmentation if set')
parser.add_argument('--architecture', default='badnet', help='Model architecture (badnet for CNN, vit for Vision Transformer)')
parser.add_argument('--no_pretrain_load', action='store_true', help='Skip loading pretrained checkpoint even if it exists')



# TODO: will add. under create_backdoor_loader
parser.add_argument('--trigger_size', type=int, default=5, help='Size of the trigger patch (e.g. 5x5)')
parser.add_argument('--trigger_position', default='corner', help='Where to place the patch (corner or center)')


opt = parser.parse_args()