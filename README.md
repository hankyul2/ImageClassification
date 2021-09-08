# Image Classification PyTorch (WIP)
This is basic Image Classification Repo. It contains pytorch implemented image classification models.



## Tutorial

1. clone repo and install requirements

   ```bash
   git clone https://github.com/hankyul2/ImageClassification.git
   pip3 install requirements.txt
   ```

   

2. download dataset (cifar10, 100 from pytorch )

   ```bash
   python3 main.py --download_dataset
   ```

   

3. train model (pretrained or not)

   ```bash
   python3 main.py -m vit_base_patch16_224 -d cifar10 --pretrained
   ```

   



## Experiment

- ResNet Architecture follow official pytorch implementation, get weight from pytorch too.

- ViT Architecture follow timm implementation(slightly different), get weight from official [google vision-transformer github](https://github.com/google-research/vision_transformer) 
- All scores are 3 times average scores

| Architecture                                                 | Pretrained on | Cifar10 | Cifar100 |
| ------------------------------------------------------------ | ------------- | ------- | -------- |
| ResNet50                                                     | ?             | 96.4    | 84.2     |
| ResNet101                                                    | ?             | 97.4    | 86.1     |
| ViT_base_16_224<br />([summary](docs/vit_base_patch16_224.md), ) | ImageNet21k   | 98.5    | 91.0     |
| ViT_base_32_224<br />([summary](docs/vit_base_patch32_224.md), ) | ImageNet21k   | 98.2    | 89.9     |
| ViT_large_16_224<br />([summary](docs/vit_large_patch16_224.md), ) | ImageNet21k   | 99.1    |          |
| ViT_large_32_224<br />([summary](docs/vit_large_patch32_224.md), ) | ImageNet21k   | 98.4    | 90.7     |
| R50 + ViT_base_16_224<br />([summary](docs/vit_base_patch16_224.md), ) | ImageNet21k   |         |          |
| R50 + ViT_large_32_224<br />([summary](docs/vit_base_patch16_224.md), ) | ImageNet21k   |         |          |



## References

1. Residual Learning
2. ViT
3. Timm
4. torchvision