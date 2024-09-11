# LoLi-IEA: low-light image enhancement algorithm

## Overview

Official PyTorch implementation of [LoLi-IEA: a low-light image enhancement algorithm](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12675/1267512/LoLi-IEA-low-light-image-enhancement-algorithm/10.1117/12.2677422.short#_=_) presented at the SPIE Optical Engineering + Applications 2023 conference, San Diego, California, United States.

![LoLi_Architecture](Architecture.png)

## Requirements

1. opencv-python == 4.9.0.80
2. scikit-image == 0.22.0
3. numpy == 1.24.3
4. torch == 2.3.0+cu118
5. Pillow == 10.2.0
6. tqdm ==  4.65.0
7. natsort == 8.4.0
8. torchvision == 0.18.0+cu118

## Inference
To test the model, follow these steps:

1. Download the [weights](https://drive.google.com/file/d/1uLIrWoW6WEqQDtYNdg-Lx3tGFlYSjavU/view?usp=sharing) for the Pretrained Model and place them in the ./Models directory.  

2. Place your images to be enhanced in the ./1_Input directory.

3. Run the code with the following command:

   ```bash
   python main.py

4. The enhanced images will be saved in the ./2_Output directory.

## Citation
If this work contributes to your research, we would appreciate it if you could cite our paper:

```bibtex
@inproceedings{perez2023loli,
  title={LoLi-IEA: low-light image enhancement algorithm},
  author={Perez-Zarate, Ezequiel and Ramos-Soto, Oscar and Rodr{\'\i}guez-Esparza, Erick and Aguilar, German},
  booktitle={SPIE Optical Engineering + Applications},
  volume={12675},
  pages={230--245},
  year={2023},
  organization={SPIE}
}
