---
typora-copy-images-to: pics
---

Cifar10/100 Train

Cifar10/100 을 학습 하는 방법에는 크게 두 가지가 있다. (내가 아는 선에서 정리)

1. 모델을 이미지 크기에 맞게 조정하는 방법 (ResNet) : 어려운 방법은 아니다. 그냥 모델을 단순화 하는 방법이다. 단순화된 모델의 구조는 [ResNet32]() 를 참고하자. 
   1. 단점 1 : 모든 모델을 이렇게 단순화 해야하는건 꽤 번거로운 일이다. 특히 ViT의 경우 어떻게 단순화 해야할지 예제도 찾기 힘들고 내가 생각해내기도 힘들다 (아마 패치의 크기를 4x4, 8x8로 만들면 될것 같다.) 
   2. 단점 2 : pretrained 된 weight를 사용할 수 없게 된다. 
2. 이미지의 크기를 늘린다 : 그냥 이미지의 크기를 224x224 로 늘리면 된다.
   1. 단점 1 : 이미지의 크기가 늘어남에 따라서 사용해야하는 gpu의 메모리, 연산 속도 또한 매우 많이 증가 한다.

이렇게 정리해봤자 두 가지 방법 모두 틀린 방법일 수 있다. 일단 오늘 내가 찾아본 바로는 이렇다는 것이다. 논문에서 혹은 공식레포에서 제시하는 성능이 나오지 않을 경우 계속해서 찾아봐야겠다.



GPU Usage 

| Model                   | 32x32 | 224x224      |
| ----------------------- | ----- | ------------ |
| ResNet50                | 2G    | 6.5G         |
| ResNet101               | 2.2G  | 8.5G         |
| PreActivation ResNet50  |       |              |
| PreActivation ResNet101 |       |              |
| ResNext50               |       |              |
| ResNext101              |       |              |
| Wide ResNet50           |       |              |
| Wide ResNet101          |       |              |
| EfficientNet_b0         |       |              |
| EfficientNet_b1         |       |              |
| ViT base 32x32          | 3G    | 4G           |
| ViT base 16x16          | 4G    | 9.9G         |
| ViT large 32x32         | 5G    | 8G           |
| ViT large 16x16         | 6G    | 15.5G (b:16) |
| R50 + ViT base 16x16    |       | 14.2G        |
| R50 + ViT large 32x32   |       | 12G          |

*ViT Huge 16x16, R50 + ViT large 16x16는 muti-gpu를 꼭 사용해야 됨 (RTX 3090 기준)*

