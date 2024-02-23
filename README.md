<div align="center">
  <h1 align="center">FG-Net: Facial Action Unit Detection with Generalizable Pyramidal Features</h1>

  <p align="center">

<a href="https://yufengyin.github.io/">
    Yufeng Yin</a>,
<a href="https://boese0601.github.io/">
    Di Chang</a>,
<a href="https://guoxiansong.github.io/homepage/index.html">
    Guoxian Song</a>,
<a href="https://ssangx.github.io/">
    Shen Sang</a>,
<a href="https://tiancheng-zhi.github.io/">
    Tiancheng Zhi</a>,
<a href="https://www.jingliu.net/">
    Jing Liu</a>,
<a href="http://linjieluo.com/">
    Linjie Luo</a>,
<a href="https://people.ict.usc.edu/~soleymani/">
    Mohammad Soleymani</a>
<br>
<a href="https://ict.usc.edu/">USC ICT</a>, ByteDance

<strong>WACV 2024</strong>
<br/>
<a href="https://arxiv.org/pdf/2308.12380.pdf">Arxiv</a>
<br/>
</p>
</div>

## Introduction

This is the official implementation of our WACV 2024 Algorithm Track paper: FG-Net: Facial Action Unit Detection with Generalizable Pyramidal Features.

FG-Net extracts feature maps from a StyleGAN2 model pre-trained on a large and diverse face image dataset. Then, these features are used to detect AUs with a Pyramid CNN Interpreter, making the training efficient and capturing essential local features. The proposed FG-Net achieves a strong generalization ability for heatmap-based AU detection thanks to the generalizable and semantic-rich features extracted from the pre-trained generative model. Extensive experiments are conducted to evaluate within- and cross-corpus AU detection with the widely-used DISFA and BP4D datasets. Compared with the state-of-the-art, the proposed method achieves superior cross-domain performance while maintaining competitive within-domain performance. In addition, FG-Net is dataefficient and achieves competitive performance even when trained on 1000 samples.

<p align="center">
  <img src="https://github.com/ihp-lab/FG-Net/blob/main/pipeline.png" width="700px" />
</p>

## Installation
Clone repo:
```
git clone https://github.com/ihp-lab/FG-Net.git
cd FG-Net
```

The code is tested with Python == 3.7, PyTorch == 1.10.1 and CUDA == 11.3 on NVIDIA Quadro RTX 8000. We recommend you to use [anaconda](https://www.anaconda.com/) to manage dependencies.

```
conda create -n fg-net python=3.7
conda activate fg-net
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudatoolkit-dev=11.3
pip install pandas
pip install tqdm
pip install -U scikit-learn
pip install opencv-python
pip install dlib
pip install imutils
```

## Data Structure
Download the [BP4D](https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [DISFA](http://mohammadmahoor.com/disfa/) dataset from the official website.

We first pre-process the input image by dlib to obatain the facial landmarks. The detected landmark are used to crop and align the face by [FFHQ-alignment](https://github.com/happy-jihye/FFHQ-Alignment). We finally use dlib again to detect the facial labdmarks for the aligned images to generate heatmaps.

You should get a dataset folder like below:

```
data
├── DISFA
│ ├── labels
│ │ └── 0
│ │ │ ├── train.csv
│ │ │ └── test.csv
│ │ ├── 1
│ │ └── 2
│ ├── aligned_images
│ └── aligned_landmarks
└── BP4D
```

## Checkpoints
StyleGAN2:
To get pytorch checkpoints for StyleGAN2 (stylegan2-ffhq-config-f.pt), check Section [Convert weight from official checkpoints](https://github.com/rosinality/stylegan2-pytorch/tree/master)

StyleGAN2 Encoder:
[pSp encoder](https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view?pli=1). Rename the pt file to `encoder.pt`.

AU detection:
[BP4D](https://drive.google.com/file/d/1mwOq1gceGTKvcxN0EJywY9VTuXvASgUZ/view?usp=sharing) and [DISFA](https://drive.google.com/file/d/1wTeqUNkhi-ONXXFJ4K6AozyOxyHTHog-/view?usp=sharing)

Put all checkpoints under the folder `/code/checkpoints`.

## Training and Within-domain Evaluation
```
cd code
CUDA_VISIBLE_DEVICES=0 python train_interpreter.py --exp experiments/bp4d_0.json
CUDA_VISIBLE_DEVICES=1 python train_interpreter.py --exp experiments/disfa_0.json
```

## Cross-domain Evaluation
```
cd code
CUDA_VISIBLE_DEVICES=0 python eval_interpreter.py --exp experiments/eval_b2d.json
CUDA_VISIBLE_DEVICES=1 python eval_interpreter.py --exp experiments/eval_d2b.json
```

## Singe image inference
```
cd code
CUDA_VISIBLE_DEVICES=0 python single_image_inference.py --exp experiments/single_image_inference_bp4d.json
CUDA_VISIBLE_DEVICES=1 python single_image_inference.py --exp experiments/single_image_inference_disfa.json
```

## License
Our code is distributed under the MIT License. See `LICENSE` file for more information.

## Citation
```
@article{yin2023fg,
  title={FG-Net: Facial Action Unit Detection with Generalizable Pyramidal Features},
  author={Yin, Yufeng and Chang, Di and Song, Guoxian and Sang, Shen and Zhi, Tiancheng and Liu, Jing and Luo, Linjie and Soleymani, Mohammad},
  journal={arXiv preprint arXiv:2308.12380},
  year={2023}
}
```

## Contact
If you have any questions, please raise an issue or email to Yufeng Yin (`yin@ict.usc.edu`or `yufengy@usc.edu`).

## Acknowledgments
Our code follows several awesome repositories. We appreciate them for making their codes available to public.

- [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release/)
- [StyleGAN2-pytorch](https://github.com/rosinality/stylegan2-pytorch/tree/master)
- [pSp](https://github.com/eladrich/pixel2style2pixel)
