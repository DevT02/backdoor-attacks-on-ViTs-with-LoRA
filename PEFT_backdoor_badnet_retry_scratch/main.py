import os
import pathlib
import torch
import random
import numpy as np
from data import PoisonedDataset, load_init_data, create_backdoor_data_loader
from models import BadNet, load_model
from utils.utils import print_model_perform, backdoor_model_trainer
from config import opt

def set_seed(seed):
    """
    Set all relevant random seeds to ensure reproducibility.
    """
    random.seed(seed)              # Python's built-in random
    np.random.seed(seed)           # NumPy
    torch.manual_seed(seed)        # PyTorch CPU random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # PyTorch GPU random seed

def main():
    print(f"Download dataset: {opt.download}")  
    if opt.seed is not None:
        print(f"Setting random seed = {opt.seed}")
        set_seed(opt.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("# --------------------------read dataset: %s --------------------------" % opt.dataset)
    train_data, test_data = load_init_data(dataname=opt.dataset, device=device, download=opt.download, dataset_path=opt.datapath)

    print("# --------------------------construct poisoned dataset--------------------------")
    train_data_loader, test_data_ori_loader, test_data_tri_loader = create_backdoor_data_loader(opt.dataset, train_data, test_data, opt.trigger_label, opt.poisoned_portion, opt.batchsize, device)

    print("# --------------------------begin training backdoor model--------------------------")
    # Use different checkpoint names for clean vs LoRA to avoid conflicts in 2-stage training
    if opt.use_lora:
        basic_model_path = "./checkpoints/%s-%s-lora.pth" % (opt.architecture, opt.dataset)
    else:
        basic_model_path = "./checkpoints/%s-%s-clean.pth" % (opt.architecture, opt.dataset)
    if opt.no_train:
        model = backdoor_model_trainer(
                dataname=opt.dataset,
                train_data_loader=train_data_loader, 
                test_data_ori_loader=test_data_ori_loader,
                test_data_tri_loader=test_data_tri_loader,
                trigger_label=opt.trigger_label,
                epoch=opt.epoch,
                batch_size=opt.batchsize,
                loss_mode=opt.loss,
                optimization=opt.optim,
                lr=opt.learning_rate,
                print_perform_every_epoch=opt.pp,
                basic_model_path= basic_model_path,
                device=device,

                use_lora=opt.use_lora,
                lora_rank=opt.lora_rank,
                lora_alpha=opt.lora_alpha,
                lora_dropout=opt.lora_dropout,
                freeze_weights=opt.freeze_weights,
                poisoned_portion=opt.poisoned_portion,
                architecture=opt.architecture,
                no_pretrain_load=opt.no_pretrain_load
                )
    else:
        model = load_model(basic_model_path, model_type=opt.architecture, input_channels=train_data_loader.dataset.channels, output_num=train_data_loader.dataset.class_num, device=device, dataname=opt.dataset)

    print("# --------------------------evaluation--------------------------")
    print("## original test data performance:")
    print_model_perform(model, test_data_ori_loader)
    print("## triggered test data performance:")
    print_model_perform(model, test_data_tri_loader)



if __name__ == "__main__":
    main()
