# Image Classification (PyTorch, Pytorch-lightning)
This is basic Image Classification Repo. It contains pytorch(pytorch-lightning) implemented image classification models.





## Tutorial

1. clone repo and install requirements

   ```bash
   git clone https://github.com/hankyul2/ImageClassification.git
   pip3 install requirements.txt
   ```

   

3. train model

   ```bash
   python3 main.py fit --config configs/finetune.yaml -m vit_base_patch16_224 -d cifar10 -g 0,
   ```

   



## Experiment

*In this work*

| Architecture                                                 | Pretrained on | Cifar10 | Cifar100 |
| ------------------------------------------------------------ | ------------- | ------- | -------- |
| ResNet50<br />([tf.dev](), [summary]())                      | ImageNet      | 97.6    | 86.0     |
| ResNet101<br />([tf.dev](), [summary]())                     | ImageNet      | 97.9    | 87.4     |
| SeNet50<br />([tf.dev](), [summary]())                       | ImageNet      | 97.3    | 85.8     |
| SeNet101<br />([tf.dev](), [summary]())                      | ImageNet      | 97.4    | 86.9     |
| MobileNet_v2<br />([tf.dev](), [summary]())                  | ImageNet      | 96.8    | 82.4     |
| MobileNet_v3_small<br />([tf.dev](), [summary]())            | ImageNet      | 95.0    | 80.1     |
| MobileNet_v3_large<br />([tf.dev](), [summary]())            | ImageNet      | 96.0    | 83.0     |
| EfficientNet_b0<br />([tf.dev](), [summary]())               | ImageNet      | 97.2    | 86.2     |
| EfficientNet_b1<br />([tf.dev](), [summary]())               | ImageNet      | 97.5    | 87.2     |
| EfficientNet_b2<br />([tf.dev](), [summary]())               | ImageNet      | 97.8    | 87.5     |
| EfficientNet_b3<br />([tf.dev](), [summary]())               | ImageNet      | 98.1    | 89.1     |
| EfficientNet_b4<br />([tf.dev](), [summary]())               | ImageNet      | 97.7    | -        |
| ViT_base_16_224<br />(([tf.dev](), [summary](docs/vit_base_patch16_224.md)) | ImageNet21k   | 98.6    | 92.0     |
| ViT_base_32_224<br />(([tf.dev](), [summary](docs/vit_base_patch32_224.md)) | ImageNet21k   | 98.2    | 89.9     |
| ViT_base_16_384<br />([tf.dev](), [summary](docs/vit_base_patch16_384.md)) | ImageNet21k   | 98.2    | 90.0     |
| ViT_base_32_384<br />([tf.dev](), [summary](docs/vit_base_patch32_224.md)) | ImageNet21k   | 98.1    | 90.1     |
| ViT_large_16_224<br />([tf.dev](), ([summary](docs/vit_large_patch16_224.md)) | ImageNet21k   | 99.1    | -        |
| ViT_large_32_224<br />([tf.dev](), ([summary](docs/vit_large_patch32_224.md)) | ImageNet21k   | 98.4    | 90.7     |
| ViT_large_16_384<br />([tf.dev](), [summary](docs/vit_large_patch16_224.md)) | ImageNet21k   | -       | -        |
| ViT_large_32_384<br />([tf.dev](), [summary](docs/vit_large_patch32_224.md)) | ImageNet21k   | -       | -        |
| R50 + ViT_base_16_224<br />([tf.dev](), [summary](docs/vit_base_patch16_224.md)) | ImageNet21k   | 97.7    | 88.8     |
| R50 + ViT_base_16_384<br />([tf.dev](), [summary](docs/vit_base_patch16_224.md)) | ImageNet21k   | -       | 92.3     |
| R50 + ViT_large_32_224<br />([tf.dev](), [summary](docs/vit_base_patch16_224.md)) | ImageNet21k   | 98.6    | -        |
| R50 + ViT_large_32_384<br />([tf.dev](), [summary](docs/vit_base_patch16_224.md)) | ImageNet21k   | -       | -        |

*Note*

- Pretrained weights are from timm or torchvision or [google vision-transformer github](https://github.com/google-research/vision_transformer) 
- All scores are 3 times average scores
- `efficientnet_v2` results is in [here](https://github.com/hankyul2/EfficientNetV2-pytorch)



## References

*pass*