from .badnet import BadNet
from .vit_lora import VisionTransformerLoRA, load_vit_model
import torch
from utils.utils import print_model_perform
from .LoRA import LoRAConfig

def load_model(model_path, model_type, input_channels, output_num, device, use_lora=False, rank=4, alpha=16.0, dropout=0.05, freeze_weights=True, dataname='mnist'):
    print("## load model from : %s" % model_path)
    lora_config = None
    if use_lora:
        lora_config = LoRAConfig(
            rank=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            freeze_weights=freeze_weights
        )

    if model_type =='cnn':
        model = MyCnn(input_channels=input_channels, output_num=output_num).to(device)
    elif model_type == 'cnn_paper':
        model = PaperCnn(input_channels, output_num).to(device)
    elif model_type == 'badnet':
        model = BadNet(input_channels, output_num, config=lora_config).to(device)
    elif model_type == 'vit':
        model = load_vit_model(dataname, config=lora_config).to(device)
    elif model_type == 'softmax':
        model = Softmax(input_channels, output_num).to(device)
    elif model_type == 'mlp':
        model = MLP(input_channels, output_num).to(device)
    elif model_type == 'lr':
        model = LogsticRegression(input_channels, output_num).to(device)
    else:
        print("can't match your input model type, please check...")

    # Load state dict, allowing for mismatched keys (e.g., when adding LoRA to pretrained model)
    try:
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"## Loaded weights from {model_path} (strict=False to allow LoRA addition)")
    except Exception as e:
        print(f"## Warning: Could not load from {model_path}: {e}")

    return model

