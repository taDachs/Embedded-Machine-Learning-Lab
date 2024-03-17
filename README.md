# Embedded Machine Learning Lab WS23

faf stands for fast-as-fuck

## TODO:

- [x] person-only detection
  - [x] transfer weights
  - [x] finetuning
- [x] batch norm inference optimization
- [x] pruning
- [x] integration
- [x] inference framework
- [ ] detection pipeline

optional:

- [ ] tensorrt
- [ ] data augmentation
- [ ] more data
- [ ] profiling
- [ ] quantization

## Datasets

- [tiktok dancing](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-tiktok-dancing-dataset)
- [human dataset](https://www.kaggle.com/datasets/fareselmenshawii/human-dataset)

## Cuda libs

```
pip3 uninstall torch torchvision
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## jetcam package

```
pip3 install git+https://github.com/NVIDIA-AI-IOT/jetcam.git
```
