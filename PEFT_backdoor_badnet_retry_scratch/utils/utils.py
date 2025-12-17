import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from models.badnet import BadNet
from models.vit_lora import load_vit_model
from models.LoRA import LoRAConfig
from datetime import datetime

def array2img(x):
    a = np.array(x)
    img = Image.fromarray(a.astype('uint8'), 'RGB')
    img.show()


def print_model_perform(model, data_loader):
    model.eval() # switch to eval mode
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    try:
        target_names_idx = set.union(set(np.array(y_true.cpu())), set(np.array(y_predict.cpu())))
        target_names = [data_loader.dataset.classes[i] for i in target_names_idx]
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=target_names))
    except ValueError as e:
        print(e)


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign MSE loss function to you...")
        criterion = nn.MSELoss()
    return criterion


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def safe_cpu_scalar(x):
    """
    If x is a GPU tensor of shape [], returns its Python float (x.item()).
    Otherwise, returns x unchanged.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().item()  # works if x is a 0-dim or single-scalar tensor
    return x

def show_graphs(lora_config, train_process, results_dir="./results", lr=None, batch_size=None, poison_ratio=None, architecture="badnet"):
    print("Plotting results...")
    # Unpack train_process
    _, _, _, _, epochs, losses, train_accs, ori_test_accs, trigger_test_accs = zip(*train_process)

    # Convert GPU tensors to Python floats if needed
    epochs = [safe_cpu_scalar(e) for e in epochs]
    losses = [safe_cpu_scalar(l) for l in losses]
    train_accs = [safe_cpu_scalar(a) for a in train_accs]
    ori_test_accs = [safe_cpu_scalar(a) for a in ori_test_accs]
    trigger_test_accs = [safe_cpu_scalar(a) for a in trigger_test_accs]

    # Now everything is a float or int, safe to plot
    plt.figure(figsize=(12, 6))

    # 1) Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()

    # 2) Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy", color="blue")
    plt.plot(epochs, ori_test_accs, label="Original Test Accuracy", color="green")
    plt.plot(epochs, trigger_test_accs, label="Trigger Test Accuracy", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracies Over Epochs")
    plt.legend()

    # LoRA config text
    lora_text = (
        f"LoRA Config:\n"
        f"Rank: {lora_config.rank}\n"
        f"Alpha: {lora_config.lora_alpha}\n"
        f"Dropout: {lora_config.lora_dropout}\n"
        f"Freeze: {lora_config.freeze_weights}"
    )


    if lr is not None:
        lora_text += f"\nLR: {lr}"
    if batch_size is not None:
        lora_text += f"\nBatch size: {batch_size}"
    if poison_ratio is not None:
        lora_text += f"\nPoison ratio: {poison_ratio}"

    plt.gcf().text(0.75, 0.3, lora_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()

    # Make sure results_dir exists
    os.makedirs(results_dir, exist_ok=True)

    # Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{architecture}_lora_r{lora_config.rank}_alpha{lora_config.lora_alpha}_{timestamp}.png"
    save_path = os.path.join(results_dir, filename)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")

def backdoor_model_trainer(
    dataname, 
    train_data_loader, 
    test_data_ori_loader, 
    test_data_tri_loader, 
    trigger_label, 
    epoch, 
    batch_size, 
    loss_mode, 
    optimization, 
    lr, 
    print_perform_every_epoch, 
    basic_model_path, 
    device,
    # LoRA hyperparams
    use_lora=True,
    lora_rank=16,
    lora_alpha=16.0,
    lora_dropout=0.05,
    freeze_weights=True,
    poisoned_portion=0.1,
    architecture='badnet',
    no_pretrain_load=False
):
    """
    If use_lora=True, we freeze base weights and only train LoRA parameters.
    Otherwise, you could adapt the code to fully train all layers. 
    """

    # Create LoRAConfig
    lora_config = LoRAConfig(
        rank=lora_rank if use_lora else 0,
        lora_alpha=lora_alpha if use_lora else 0.0,
        lora_dropout=lora_dropout if use_lora else 0.0,
        freeze_weights=freeze_weights
    ) if use_lora else None

    # Initialize model based on architecture
    if architecture == 'vit':
        badnet = load_vit_model(dataname, config=lora_config).to(device)
    else:
        badnet = BadNet(
            input_channels=train_data_loader.dataset.channels,
            output_num=train_data_loader.dataset.class_num,
            config=lora_config
        ).to(device)

    # Load pretrained checkpoint if exists (for 2-stage training)
    # For LoRA runs, try to load clean checkpoint first
    pretrain_path = basic_model_path
    if use_lora and architecture == 'vit':
        clean_path = basic_model_path.replace('-lora.pth', '-clean.pth')
        if os.path.exists(clean_path):
            pretrain_path = clean_path
    
    if os.path.exists(pretrain_path) and not no_pretrain_load:
        try:
            state_dict = torch.load(pretrain_path)
            # Use strict=False to allow adding LoRA layers on top of pretrained base
            missing_keys, unexpected_keys = badnet.load_state_dict(state_dict, strict=False)
            print(f"## Loaded pretrained weights from {pretrain_path}")
            if missing_keys:
                # Compute actual number of trainable parameters
                num_params = sum(p.numel() for name, p in badnet.named_parameters() 
                                if any(k in name for k in missing_keys) and p.requires_grad)
                print(f"## Missing keys (LoRA layers): {len(missing_keys)} tensors ({num_params:,} parameters)")
            if unexpected_keys:
                print(f"## Warning: Unexpected keys found (will be ignored): {len(unexpected_keys)} tensors")
        except Exception as e:
            print(f"## Warning: Could not load pretrained weights: {e}")
            print(f"## Starting from random initialization")
    elif os.path.exists(pretrain_path) and no_pretrain_load:
        print(f"## Checkpoint exists but --no_pretrain_load specified, starting fresh")
    else:
        print(f"## No pretrained checkpoint found, starting from random initialization")

    # 1) Define the loss function here
    criterion = loss_picker(loss_mode)

    # 2) Pick optimizer (as you already do)
    if use_lora:
        lora_parameters = [p for p in badnet.parameters() if p.requires_grad]
        optimizer = optimizer_picker(optimization, lora_parameters, lr=lr)
    else:
        optimizer = optimizer_picker(optimization, badnet.parameters(), lr=lr)



    train_process = []
    print(f"### target label={trigger_label}, EPOCH={epoch}, LR={lr}, use_lora={use_lora}")
    print(f"### Train size={len(train_data_loader.dataset)}, "
          f"ori test size={len(test_data_ori_loader.dataset)}, "
          f"tri test size={len(test_data_tri_loader.dataset)}\n")

    try:
        for epo in range(epoch):
            running_loss = train(badnet, train_data_loader, criterion, optimizer, loss_mode)
            acc_train = eval(badnet, train_data_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
            acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)
            acc_test_tri = eval(badnet, test_data_tri_loader, batch_size=batch_size, mode='backdoor', print_perform=print_perform_every_epoch)

            print(f"# EPOCH {epo}   loss: {running_loss:.4f}, "
                  f"train acc: {acc_train:.4f}, "
                  f"ori test acc: {acc_test_ori:.4f}, "
                  f"trigger test acc: {acc_test_tri:.4f}\n")

            # (Optional) If you want to do post-training quantization, 
            # do it after training is done, not every epoch. Otherwise comment out.
            # torch.quantization.convert(badnet, inplace=True)

            # Save model
            torch.save(badnet.state_dict(), basic_model_path)

            # Save training progress
            train_process.append((dataname, batch_size, trigger_label, lr, epo, running_loss, acc_train, acc_test_ori, acc_test_tri))
            df = pd.DataFrame(train_process, 
                              columns=["dataname", "batch_size", "trigger_label", "learning_rate", "epoch", "loss", "train_acc", "test_ori_acc", "test_tri_acc"])
            # Example: store in a unique CSV name
            csv_filename = f"./logs/{architecture}_{dataname}_loraRank{lora_rank}_alpha{lora_alpha}_trigger{trigger_label}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
    except KeyboardInterrupt:
        print("Training interrupted. Plotting partial results...")
        if lora_config is not None:
            show_graphs(
                lora_config=lora_config,
                train_process=train_process,
                results_dir=f"./results/{dataname}",
                lr=lr,
                batch_size=batch_size,
                poison_ratio=poisoned_portion,
                architecture=architecture
            )
        return badnet

    if lora_config is not None:
        show_graphs(
            lora_config=lora_config,
            train_process=train_process,
            results_dir=f"./results/{dataname}",
            lr=lr,
            batch_size=batch_size,
            poison_ratio=poisoned_portion,
            architecture=architecture
        )

    return badnet



def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        
        batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if print_perform and mode != 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return accuracy_score(y_true.cpu(), y_predict.cpu())