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

## Installation
TODO

## Training and Within-domain Evaluation
TODO

## Cross-domain Evaluation
TODO

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
- [pSp](https://github.com/eladrich/pixel2style2pixel)
- [XNorm](https://github.com/ihp-lab/XNorm)
