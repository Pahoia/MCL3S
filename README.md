# Multi-layer Feature Fusion and Coarse-to-fine Label Learning for Semi-supervised Lesion Segmentation of Lung Cancer
## Introduction
Code for our paper: 'Multi-layer Feature Fusion and Coarse-to-fine Label Learning for Semi-supervised Lesion Segmentation of Lung Cancer'
## Usage
1.Train;
```
python code/train_MCL3S.py --model vnet --labelnum 10 --gpu 0 --seed 1337
```
2.Test;
```
python code/test_3D.py --model vnet --gpu 0
```

## Acknowledgements
Thanks for these authors [ST++](https://github.com/LiheYoung/ST-PlusPlus), [MC-Net+](https://github.com/ycwu1997/MC-Net), [SSL4MIS](https://github.com/HiLab-git/SSl4MIS) for their valuable works.

## Contact
If you have any question, please contact pahoia_boho651@outlook.com.

## Citation
If you find our code is useful, please consider citing our paper as follows:
```
@article{chen2025multi,
  title={Multi-layer feature fusion and coarse-to-fine label learning for semi-supervised lesion segmentation of lung cancer},
  author={Chen, Jiale and Feng, Siyang and Cui, Yanfen and Fan, Chuansong and Lin, Huan and Bian, Xinjun and Li, Lingqiao and Liu, Zhenbing and Liu, Zaiyi and Lan, Rushi and others},
  journal={Knowledge-Based Systems},
  pages={113451},
  year={2025},
  publisher={Elsevier}
}
```
