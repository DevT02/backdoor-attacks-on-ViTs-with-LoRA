# Backdoor Vulnerabilities in LoRA-adapted Vision Models

This repository demonstrates backdoor attack vulnerabilities in Parameter-Efficient Fine-Tuning (PEFT) methods, specifically LoRA-based attacks on CNNs and Vision Transformers.

**TL;DR:**
- Backdoor attacks succeed on LoRA-adapted models with minimal parameter updates
- CNN baseline: 98.3% clean accuracy, 99.9% attack success rate (MNIST)
- Vision Transformer: 80.3% clean accuracy, 99.8% attack success rate (MNIST)
- Security concern for organizations deploying PEFT-tuned models

---

## Summary

I extended BadNets (Gu et al., 2017) to work with LoRA-style parameter-efficient fine-tuning on vision models. Standard backdoor methods assume full model retraining; this work shows that poisoning only low-rank adapters (rank 4-16) is sufficient to achieve high attack success rates.

**Key findings:**
1. LoRA adapters alone can implant effective backdoors without modifying base weights
2. Attack Success Rate reaches 95-100% within 7-15 epochs with 10% poisoned data
3. Clean accuracy remains high (80-98% depending on architecture)
4. Poisoned models appear normal under standard evaluation metrics

**Architectures tested:**
- CNN with LoRA on convolutional and linear layers (complete)
- Vision Transformer with LoRA on attention and MLP layers (complete)

---

## Environment Setup

**Create conda environment:**
```bash
conda create -n backdoor-lora python=3.9 -y
conda activate backdoor-lora
```

**Install PyTorch with CUDA:**
```bash
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**Install dependencies:**
```bash
pip install tqdm matplotlib pandas numpy scikit-learn
```

Or for CPU-only:
```bash
conda install pytorch=2.0.1 torchvision=0.15.2 cpuonly -c pytorch -y
```

---

## Quick Start

### CNN Baseline (5 minutes on GPU)

```bash
cd PEFT_backdoor_badnet_retry_scratch
python main.py --dataset mnist --architecture badnet --use_lora --lora_rank 8 --lora_alpha 8.0 --lora_dropout 0.1 --freeze_weights --epoch 50 --learning_rate 0.001 --batchsize 64 --poisoned_portion 0.1 --seed 42
```

Expected results: ~98% clean accuracy, ~99.9% attack success rate

### Vision Transformer (2-stage, 30 min on GPU)

**Stage 1: Pre-train clean ViT**
```bash
python main.py --dataset mnist --architecture vit --epoch 30 --learning_rate 0.001 --batchsize 64 --poisoned_portion 0.0 --seed 42 --loss cross --optim adam
```

**Stage 2: Add LoRA + backdoor**
```bash
python main.py --dataset mnist --architecture vit --use_lora --lora_rank 4 --lora_alpha 4.0 --lora_dropout 0.1 --freeze_weights --epoch 50 --learning_rate 0.0001 --batchsize 64 --poisoned_portion 0.1 --seed 42 --loss cross --optim adam
```

Expected results: ~80% clean accuracy, ~99.8% attack success rate

---

## Key Results (MNIST)

| Architecture | LoRA Config | Clean Acc | Attack Success Rate | Epochs to 90% ASR |
|-------------|-------------|-----------|---------------------|-------------------|
| CNN | rank=8, alpha=8.0 | 98.3% | 99.9% | ~8-12 |
| ViT (2-stage) | rank=4, alpha=4.0 | 80.3% | 99.8% | ~7-10 |

**Note:** ViT requires 2-stage training (pre-train clean model at 76% accuracy, then add LoRA + backdoor while freezing base weights). CNN can be trained end-to-end. The ViT result demonstrates that LoRA adapters alone (49,704 trainable parameters) can inject a backdoor into a frozen pre-trained model while improving clean accuracy from 76% to 80%.

More detailed MNIST results showing high attack success rates across different configurations are available on [Google Drive](https://drive.google.com/drive/u/1/folders/1IEs7MTKmkxjcDRGOCzlvCRgtcoa21771).

---

## Full Hyperparameter Sweep

To reproduce the full hyperparameter exploration (runs 1000+ experiments):

```bash
cd PEFT_backdoor_badnet_retry_scratch
python multi_experiment_another_try.py
```

This tests combinations of:
- Architectures: badnet (CNN), vit
- LoRA ranks: 4, 8, 16
- LoRA alphas: 1.0, 8.0, 16.0
- Dropout: 0.01, 0.1, 0.35
- Poison ratios: 0.01, 0.05, 0.1, 0.15
- Learning rates: 0.0001, 0.001, 0.01
- Batch sizes: 32, 64, 128

Results are saved to `logs/` (CSV) and `results/` (plots).

---

## Project Structure

The main implementation is in `PEFT_backdoor_badnet_retry_scratch/`:
- `models/badnet.py` - CNN with optional LoRA layers
- `models/vit_lora.py` - Vision Transformer with optional LoRA
- `models/LoRA.py` - LoRA layer implementations (LoRAConv2d, LoRALinear)
- `data.py` - Dataset loading and trigger injection
- `main.py` - Single experiment runner
- `multi_experiment_another_try.py` - Hyperparameter sweep
- `config.py` - Command-line arguments
- `utils/utils.py` - Training loop and evaluation

Other directories:
- `peft_lora_for_the_hpcc/` - Archived duplicate (see `ARCHIVED.md`)
- `VPT_Backdoor_VIT_Tiny/` - Exploratory Visual Prompt Tuning experiments (incomplete)
- `tdc2023-starter-kit/` - Unrelated starter code

---

## Limitations

**CIFAR-10:** Experiments are ongoing; hyperparameter tuning has not yet achieved the same strong results as MNIST. Current best attempts show lower clean accuracy (~60-70%) and less stable ASR.

**Defense mechanisms:** This work focuses on attack feasibility; no defense strategies are implemented. Future work could explore detection methods or certified defenses.

**Trigger pattern:** Uses a simple 5x5 white square patch; more sophisticated triggers (blended, semantic) are not tested.

**Model scale:** Experiments use small models (custom CNN, lightweight ViT); larger pre-trained foundation models may behave differently.

---

## Responsible Use

This is research code for evaluating robustness of PEFT methods. Do not use this to deploy malicious backdoors. Only run experiments in authorized environments with your own data and models.

---

## Attribution

Created and maintained by Devansh Tayal ([@DevT02](https://github.com/DevT02)).

Based on:
- Gu et al. (2017), *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain*
- Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*

## Citation

If you use this code or results, please cite:

```bibtex
@misc{tayal2025backdoor_lora,
  author = {Tayal, Devansh},
  title = {Backdoor Vulnerabilities in LoRA-adapted Vision Models},
  year = {2025},
  url = {https://github.com/DevT02/backdoor-attacks-on-ViTs-with-LoRA},
  note = {GitHub repository},
}
```
_Last verified: December 2025_
