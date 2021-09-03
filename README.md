# Image Classification (pytorch)
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

ResNet Architecture follow official pytorch implementation, get weight from pytorch too.

ViT Architecture follow timm implementation(slightly different), get weight from official [google vision-transformer github](https://github.com/google-research/vision_transformer) 

| Architecture | Pretrained on |      |
| ------------ | ------------- | ---- |
| ResNet       | ?             |      |
| ViT          | ImageNet21k   |      |
| R50+ViT      | ImageNet21k   |      |



## References

1. Residual Learning
2. ViT
3. Timm
4. torchvision