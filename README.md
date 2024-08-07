# LoLi-IEA: low-light image enhancement algorithm

## Overview

This repository contains the source code and associated materials for the paper titled **LoLi-IEA: low-light image enhancement algorithm**. The aim of this research is the visual enhancement of low-light images.

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

## Test
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
  booktitle={Applications of Machine Learning 2023},
  volume={12675},
  pages={230--245},
  year={2023},
  organization={SPIE}
}
