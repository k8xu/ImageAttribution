# Detecting Image Attribution for Text-to-Image Diffusion Models in RGB and Beyond
### [Paper](https://arxiv.org/pdf/2403.19653.pdf) | [arXiv](https://arxiv.org/abs/2403.19653) | [Demo](#demo) | [Bibtex](#bibtex)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oYePJ_zeV3znBE_co06mVcFNZZ5Rbb2D?usp=sharing)

[Katherine Xu](https://k8xu.github.io)$^{1}$, [Lingzhi Zhang](https://owenzlz.github.io)$^{2}$, [Jianbo Shi](https://www.cis.upenn.edu/~jshi)$^1$<br>
$^1$ University of Pennsylvania, $^2$ Adobe Inc.

## 🚀 Updates
- **4/23/2024:** Released a demo using EfficientFormer

## Setup
- Clone this repo
```
git clone https://github.com/k8xu/ImageAttribution.git
```

- Install dependencies
```
conda create --name attribution python=3.10 -y
conda activate attribution
pip install opencv-python torch pillow
```

<a name="demo"></a>
## Demo

Download a torchscript checkpoint and place it under the specified folder.
| Model Name | Torchscript | Folder | Test Accuracy |
|:----------:|:-----------:|:------:|:-------------:|
| efficientformer | [efficientformer_torchscript](https://drive.google.com/file/d/1lhyMC5DcpjrT4bocCLlGmAshRWCvsbi0/view?usp=sharing) (118M) | ./deployment/efficientformer | 90.03% |

We randomly sampled 10 test images per class, so you can quickly try our image attributor. Please check out `./images`.
```
python demo.py --img {IMAGE PATH}
```

<a name="bibtex"></a>
## Citation

If you find our work useful, please cite our paper:
```
@article{xu2024detecting,
    title={Detecting Image Attribution for Text-to-Image Diffusion Models in RGB and Beyond},
    author={Xu, Katherine and Zhang, Lingzhi and Shi, Jianbo},
    journal={arXiv preprint arXiv:2403.19653},
    year={2024}
}
```
