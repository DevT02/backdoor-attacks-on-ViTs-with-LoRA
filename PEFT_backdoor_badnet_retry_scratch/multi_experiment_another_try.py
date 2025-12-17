import os
import itertools
import subprocess

##########################
# Hyperparameter grids
##########################
architectures = ['badnet', 'vit']  # Test both CNN and ViT
ranks = [4, 8, 16]           # LoRA rank
alphas = [1.0, 8.0, 16.0]    # LoRA alpha
dropouts = [0.01, 0.1, 0.35]    
poison_ratios = [0.01, 0.05, 0.1, 0.15]  # Poison portion
epochs_list = [100]      # Number of epochs (at least 50, as requested)
lr_list = [0.01, 0.001, 0.0001]    # Multiple learning rates
batch_sizes = [32, 64, 128]  # Add varying batch sizes

# If you don’t need or want seeds, remove or comment out
SEED = 42

##########################
# Additional arguments
##########################
DATASET = "cifar10"
DATAPATH = "./dataset"
TRIGGER_SIZE = 5
DOWNLOAD = True
USE_LORA = True
FREEZE_WEIGHTS = True   # typical LoRA usage: freeze base weights
DATA_AUG = True         # whether you do data augmentation, etc.

# Create checkpoint directory if needed
os.makedirs("./checkpoints", exist_ok=True)

##########################
# Tracking Completed Combinations
##########################
completed_file = "completed_combinations.txt"

# load + skip completed combinations
if os.path.exists(completed_file):
    with open(completed_file, "r") as file:
        completed_combinations = set(file.read().splitlines())
else:
    completed_combinations = set()

def save_completed_combination(combo):
    """Save a completed combination to the file."""
    with open(completed_file, "a") as file:
        file.write(combo + "\n")

##########################
# Main loop
##########################
for arch, r, alpha, p_ratio, dropout, epochs, lr_val, batch_size in itertools.product(
    architectures, ranks, alphas, poison_ratios, dropouts, epochs_list, lr_list, batch_sizes
):
    # ex name badnet-cifar10_r8_alpha16.0_p0.05_e50_lr0.001_bs64.pth
    checkpoint_name = (
        f"{arch}-{DATASET}"
        f"_r{r}"
        f"_alpha{alpha}"
        f"_p{p_ratio}"
        f"_d{dropout}"
        f"_e{epochs}"
        f"_lr{lr_val}"
        f"_bs{batch_size}.pth"
    )

    # checkpoint_name is the unique identifier, skip if already done
    unique_id = checkpoint_name
    if unique_id in completed_combinations:
        print(f"Skipping already completed: {unique_id}")
        continue

    cmd_list = [
        "python", "main.py",
        f"--dataset {DATASET}",
        f"--datapath {DATAPATH}",
        f"--architecture {arch}",
        f"--batchsize {batch_size}",
        f"--poisoned_portion {p_ratio}",
        f"--trigger_size {TRIGGER_SIZE}",
        f"--lora_rank {r}",
        f"--lora_alpha {alpha}",
        f"--lora_dropout {dropout}",
        f"--epoch {epochs}",
        f"--learning_rate {lr_val}",
        f"--checkpoint_name {checkpoint_name}",
        "--use_lora" if USE_LORA else "",
        "--freeze_weights" if FREEZE_WEIGHTS else "",
        "--data_aug" if DATA_AUG else ""
    ]

    # If seed necessary
    cmd_list.append(f"--seed {SEED}")

    if DOWNLOAD:
        cmd_list.append("--download")

    cmd_list = [arg for arg in cmd_list if arg]
    cmd_str = " ".join(cmd_list)
    print(f"Running: {cmd_str}")
    result = subprocess.run(cmd_str, shell=True)
    if result.returncode == 0:
        save_completed_combination(unique_id)
    else:
        print(f"Error occurred for: {unique_id}")
        break  
    