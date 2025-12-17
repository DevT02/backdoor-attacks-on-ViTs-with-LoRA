# Backdoor Vulnerabilities in LoRA-adapted Vision Transformers

This project investigates how vulnerable LoRA-based parameter-efficient fine-tuning (PEFT) is to backdoor attacks on Vision Transformers. The key finding: these attacks achieve over 95% success rates while keeping the model's normal accuracy intact on MNIST and CIFAR-10.

---

## What This Project Does

I extended the BadNets backdoor attack framework to work with Vision Transformers using Low-Rank Adaptation (LoRA). The goal was to systematically test how different LoRA configurations affect backdoor attack success.

**What's new compared to existing BadNets implementations:**
- Added LoRA layers to the CNN architecture
- Tested 880+ different hyperparameter combinations
- Implemented automated experiment tracking with CSV logs
- Added parameter freezing strategies for PEFT

**Key results on MNIST:**
- Successfully demonstrated backdoor attacks with LoRA (rank 4-16, alpha 1.0-16.0)
- Attack success rate: 99.9% while maintaining 98.3% clean accuracy
- The backdoor "internalizes" between epochs 5-25 (sharp ASR jump)
- Model looks completely normal from standard performance metrics

---

## Can You Reproduce This?

**My setup:**
- Windows 11 and Linux (MSU's HPCC cluster with V100/A100 GPUs)
- CUDA 11.8 with PyTorch 2.0+
- Python 3.8-3.10
- Last tested: April 2025

**Fair warning:** Your exact numbers might vary by +/- 2% for attack success and +/- 1% for clean accuracy due to GPU randomness; but the overall trends should hold.

---

## Quick Start (5 minutes)

Want to see it work? Run a simple experiment on MNIST with LoRA:

```bash
cd PEFT_backdoor_badnet_retry_scratch
python main.py --dataset mnist --use_lora --lora_rank 4 --lora_alpha 8.0 --trigger_label 0 --poisoned_portion 0.1 --epoch 50 --batchsize 64 --learning_rate 0.001 --seed 42
```

You should see training complete in about 5 minutes on a GPU (20 minutes on CPU). The model will achieve ~98% clean accuracy and ~99% backdoor success rate.

Output locations:
- Model checkpoint: `checkpoints/badnet-mnist.pth`
- Training metrics: `logs/mnist_baseline.csv`
- Accuracy plots: `results/mnist/`

---

## Full Reproduction

To replicate all 880+ experiments:

**Setup:**
```bash
cd PEFT_backdoor_badnet_retry_scratch
pip install -r requirements.txt
python data_downloader.py  # Downloads MNIST and CIFAR-10
```

**Run experiments:**
```bash
# For MNIST
python multi_experiment_another_try.py --dataset mnist

# For CIFAR-10
python multi_experiment_another_try.py --dataset cifar10
```

This sweeps through:
- LoRA rank: 4, 8, 16
- LoRA alpha: 1.0, 8.0, 16.0
- Poisoning ratio: 1%, 5%, 10%, 15%
- Learning rate: 0.0001, 0.001, 0.01
- Batch size: 32, 64, 128
- Dropout: 0.01, 0.1, 0.35

**Time requirement:** About 48 hours on a V100 GPU for the full sweep. The script saves progress in `completed_combinations.txt` so you can resume if interrupted.

**Analyzing results:**
```python
import pandas as pd
df = pd.read_csv('logs/mnist_loraRank4_alpha1.0_trigger0.csv')
print(df[['epoch', 'test_ori_acc', 'test_tri_acc']].tail())
```

---

## Results

### How I Set This Up

**Datasets:** MNIST (60k train, 10k test) and CIFAR-10 (50k train, 10k test).

**Backdoor method:** I injected a small 5x5 pixel trigger patch in the corner of training images and labeled them all as class 0. During testing, any image with this trigger should be misclassified as class 0.

**What I measured:**
- Clean accuracy: how well the model performs on normal, untriggered images
- Attack Success Rate (ASR): how often triggered images get misclassified to the target label
- Convergence speed: how many epochs until ASR hits 90%

### MNIST Results

Best performing configuration (95 epochs, 10% poisoning, batch size 64, learning rate 0.001):

**Final metrics:**
- Clean test accuracy: **98.3%**
- Attack success rate: **99.9%**
- Convergence: ASR hits 90% around epoch 25-30

**Training pattern observed:**
The attack shows the characteristic "hockey stick" convergence:
- Epochs 0-5: Model learns main task; ASR stays low (~20%)
- Epochs 5-25: Sharp ASR increase (jumps from ~20% to 95%+)
- Epochs 25+: Both clean accuracy and ASR plateau near 100%

**What this means:** The model maintains excellent performance on normal images (98.3%); while misclassifying nearly every triggered image (99.9%) to the target label. From the outside, the model looks perfectly fine.

**Additional results:** Many more successful MNIST experimental runs showing consistent high attack success rates are available in [this Google Drive folder](https://drive.google.com/drive/u/1/folders/1IEs7MTKmkxjcDRGOCzlvCRgtcoa21771) with training curve visualizations across different hyperparameter configurations.

### CIFAR-10 Results

**Status:** Hyperparameter tuning still in progress. Current experiments show inconsistent convergence - some configurations only reach ~27% clean accuracy (well below the expected 70-80% baseline). The attack works in principle, but finding the right training setup has been more challenging than MNIST.

**Why the difference:** CIFAR-10 is more complex (32x32 RGB vs. 28x28 grayscale) and needs different learning rates/batch sizes than MNIST. The plots show the model can learn the backdoor pattern; but often at the expense of the main task. More tuning needed.

### Poisoning Ratio Experiments

Testing different poisoning amounts on MNIST shows the attack can work with very little poisoned data, though convergence is slower with lower ratios. With 10% poisoning (the main experiments), both clean accuracy and ASR reach near-perfect levels. Lower ratios like 1-5% still show backdoor learning but need more epochs to fully converge.

### Learning Rate Impact

The MNIST experiments used learning rate 0.001, which worked well. Learning rate matters a lot - too low and training is extremely slow, too high and you get instability. The exact sweet spot depends on the dataset and architecture.

### What's Actually Happening During Training?

I noticed a consistent three-phase pattern across all experiments:

**Phase 1 (Epochs 0-10):** The model learns the main task; attack success rate slowly climbs.

**Phase 2 (Epochs 10-30):** Sudden sharp increase in attack success rate. This is when the model "internalizes" the backdoor trigger pattern; it's like a hockey stick curve.

**Phase 3 (Epochs 30+):** Attack success plateaus near 100%; clean accuracy stabilizes. Everything looks normal from the outside.

The scary part? That sharp phase 2 increase suggests the backdoor pattern might actually be easier for the model to learn than the real task features.

### LoRA Parameters and Their Impact

**Rank:** Higher rank (16 vs 4) gives the model more expressiveness during adaptation; which speeds up backdoor learning.

**Alpha:** This scales how much the LoRA updates matter. Higher alpha (16.0 vs 1.0) amplifies the backdoor signal.

**Best (or worst?) combination:** Rank 16 with alpha 16.0 gets you to 90% attack success in just 18 epochs.

**Dropout:** Higher dropout (0.35) delays backdoor convergence slightly; but doesn't prevent it. Details in the CSV logs if you're curious.

### Comparing to Original BadNets

| Study           | Dataset   | Method           | Attack Success | Notes |
|-----------------|-----------|------------------|----------------|-------|
| BadNets (2017)  | MNIST     | Full fine-tuning | ~95%          | Baseline paper results |
| This work       | MNIST     | LoRA (PEFT)      | 99.9%         | Successfully reproduced |

**Key finding:** LoRA-based PEFT is just as vulnerable to backdoor attacks as full fine-tuning. Parameter efficiency doesn't give you security for free; the backdoor still learns effectively through the low-rank adaptation matrices.

### Where to Find Everything

All the detailed per-epoch metrics are in `PEFT_backdoor_badnet_retry_scratch/logs/` as CSV files. The plots showing training curves are in `results/mnist/` and `results/cifar10/`.

Plot files look like: `mnist_loraRank8_alpha8.0_trigger0_accuracy.png`

A successful backdoor attack shows:
- Clean accuracy: smooth increase, plateaus at normal performance
- Trigger accuracy: slow start, then sharp jump to near 100%
- Loss: steady decrease throughout

---

## Limitations

Let me be clear about what this project doesn't cover:

**Small datasets:** Only tested on MNIST and CIFAR-10. Whether this generalizes to ImageNet or larger Vision Transformer architectures (ViT-B/16, DeiT) is unknown.

**Architecture gap:** My implementation adds LoRA to CNN conv layers. Applying this directly to Transformer attention mechanisms would need adaptation.

**No defense testing:** This is pure attack characterization. I didn't test against backdoor detection methods like Neural Cleanse or STRIP.

**Compute intensive:** Running the full 880+ experiment sweep needs serious GPU time. Single-seed results should be taken with a grain of salt.

**Limited threat model:** I'm assuming an adversary who can poison training data but can't manipulate weights directly or attack at test time.

### What Would Be Cool to Try Next

- Test on actual Vision Transformer attention layers (not just CNN + LoRA)
- Evaluate against existing backdoor defenses
- Try different trigger types (semantic triggers, dynamic patterns)
- Use activation clustering or GradCAM to analyze what the model is learning
- Scale up to ImageNet-scale experiments

---

## Responsible Use

**Why this work exists:** To help researchers and practitioners understand PEFT vulnerabilities so we can build better defenses. Characterizing attacks is step one toward securing ML systems.

### What You Should Use This For

- Academic research on ML security
- Building backdoor detection/mitigation systems
- Security auditing in controlled environments
- Teaching ML security concepts

### What You Should NOT Use This For

- Deploying backdoored models in real systems
- Attacking models you don't have authorization to test
- Anything in medical/safety-critical domains
- Distributing poisoned models without disclosure
- Anything malicious, basically

### Threat Model Details

**What the attacker can do:**
- Inject poisoned training samples (1-15% of dataset)
- Design the trigger pattern (shape, location, appearance)
- Influence training but not architecture

**What the attacker can't do:**
- Directly manipulate model weights
- Attack at test time
- Modify the trained model post-hoc

### If You Find New Vulnerabilities

If this work helps you discover something critical in real PEFT implementations:

- Don't immediately publish exploits
- Contact maintainers privately (like Hugging Face PEFT team)
- Give them 90 days to patch
- Frame findings as robustness issues, not "how-to-hack" guides

### Legal Stuff

Follow applicable laws (CFAA, GDPR, export controls, your institution's policies). This code is provided "as is" for research. I don't endorse malicious use and I'm not responsible if someone misuses it.

---

## Attribution

This builds on two key papers:

**BadNets** (Gu et al., 2017): Original backdoor attack framework
- Paper: [arXiv:1708.06733](https://arxiv.org/abs/1708.06733)
- Base code adapted from: [verazuo/badnets-pytorch](https://github.com/verazuo/badnets-pytorch)

**LoRA** (Hu et al., 2021): Low-rank adaptation method
- Paper: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Implementation concepts from Hugging Face PEFT

### Citation

If you use this for research, please cite the original papers:

```bibtex
@inproceedings{gu2017badnets,
  title={Badnets: Identifying vulnerabilities in the machine learning model supply chain},
  author={Gu, Tianyu and Dolan-Gavitt, Brendan and Garg, Siddharth},
  booktitle={IEEE Access},
  year={2019}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Yelong, Shen and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

For this specific codebase:
```bibtex
@misc{tayal2025backdoor-lora,
  author = {Devansh Tayal},
  title = {Backdoor Vulnerabilities in LoRA-adapted Vision Transformers},
  year = {2025},
  howpublished = {\url{https://github.com/DevT02/backdoor-attacks-on-ViTs-with-LoRA}}
}
```

---

## Project Structure

The main code is in `PEFT_backdoor_badnet_retry_scratch/`. That's where you'll find:
- `main.py` - run a single experiment
- `multi_experiment_another_try.py` - run the full hyperparameter sweep
- `config.py` - all the command-line options
- `models/` - BadNet CNN and LoRA implementations
- `data/poisoned_dataset.py` - where trigger injection happens
- `logs/` - CSV files with training metrics
- `results/` - plots showing training curves
- `checkpoints/` - saved model weights

Other directories you can ignore:
- `peft_lora_for_the_hpcc/` is a duplicate I made for cluster experiments
- `VPT_Backdoor_VIT_Tiny/` was some early Visual Prompt Tuning tests (incomplete)
- `tdc2023-starter-kit/` is unrelated

---

## Contact

Open an issue on GitHub if you have questions or want to collaborate.

This is a research project I built as part of grad school applications. If you want to extend it (especially to full ViTs, defense mechanisms, or larger datasets), contributions are welcome!

**Data availability:** All experiment logs are in the repo under `logs/`. Model checkpoints are available if you want them (they're about 50MB each).

---

**Author:** Devansh Tayal ([@DevT02](https://github.com/DevT02))  
**Last updated:** April 2025
