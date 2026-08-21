# Vision Transformer Hashing for HSI 

Vision Transformer Hashing (VTS) utilizes the Vision Transformer to generate the hash code for image retrieval. 
It is tested under different retrieval frameworks such as DSH, HashNet, GreedyHash, IDHN, CSQ and DPN.



<img width="1328" height="806" alt="image" src="https://github.com/user-attachments/assets/11a21848-66b5-4ca2-b55d-cd3eb3bc79ba" />




## How to Run

This code uses the Vision Transformer (ViT) code and pretrained model (https://github.com/jeonsworld/ViT-pytorch) and DeepHash framework (https://github.com/swuxyj/DeepHash-pytorch).

Download the ViT pretrained models from official repository and keep under pretrainedVIT directory:

ViT-B_16: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz 

ViT-B_32: https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz

Download data from https://github.com/swuxyj/DeepHash-pytorch for different dataset, if not already present under data directory.

