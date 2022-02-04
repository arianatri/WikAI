<h1><center> Computer Vision Reference </center></h1>
<hr/>

## Table of Contents

- [Frameworks](#Frameworks)
  * [Image processing libraries](#Image-processing-libraries)
  * [Image Augmentation](#Image-Augmentation)
  * [Low-level Deep Learning Frameworks](#Low-level-Deep-Learning-Frameworks)
    + [<img src="https://pytorch.org/favicon.ico" width="32" />PyTorch](#-Pytorch)
    + [<img src="https://www.tensorflow.org/favicon.ico" width="32" />TensorFlow](#-TensorFlow)
    + [<img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/>Caffe](#-Caffe)
    + [<img src="https://pjreddie.com/static/icon.png" width="32"/>Darknet](#-Darknet)
    + [<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png" width="32"/>MXNet](#-MXNet)
  * [High-level frameworks](#High-level-frameworks)
    + [<img src="https://pytorch.org/favicon.ico" width="32" />Torchvision](#-Torchvision)
    + [<img src="https://assets.website-files.com/5f76c986da6f6011315a6c45/5f7b485594e5335f56984a3c_Lightning_FavIcon.png" width="32">PyTorch Lightning](#-PyTorch-Lightning)
    + [<img src="https://keras.io/favicon.ico" width="32">Keras](#-Keras)
    + [<img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/>GluonCV](#-GluonCV)
    + [<img src="https://mediapipe.dev/assets/img/favicon.svg" width="32">MediaPipe](#-MediaPipe)
    + [<img src="https://production-media.paperswithcode.com/libraries/dete.png" width="32">Detectron2](#-Detectron2)
    + [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>MMCV](#-MMCV)
- [Tasks](#Tasks)
  * [:camera: Image classification](#camera-image-classification)
  * [:mag: Object detection](#mag-object-detection)
  * [:busts_in_silhouette: Image Segmentation](#busts_in_silhouette-image-segmentation)
  * [:pushpin: Landmark/Keypoint Extraction](#pushpin-landmarkkeypoint-extraction)
  * [:triangular_ruler: Metric Learning / Few-Shot Learning](#triangular_ruler-metric-learning--few-shot-learning)
  * [:bookmark_tabs: OCR](#bookmark_tabs-OCR)
  * [:art: Image synthesis](#art-Image-synthesis)
  * [:paintbrush: Image Editting](#paintbrush-Image-Editing)
- [Datasets](#Datasets)
  * [Digit Recognition](#Digit-Recognition)
  * [CIFAR-10](#CIFAR-10)
  * [CIFAR-100](#CIFAR-100)
  * [Caltech 101](#Caltech-101)
  * [CelebA](#CelebA)
  * [CelebAMask-HQ](#CelebAMask-HQ)
  * [ImageNet](#ImageNet)
  * [COCO](#COCO)
  * [CityScapes](#CityScapes)
  * [Pascal VOC](#Pascal-VOC)
  * [CUB-200-2011](#CUB-200-2011)
  * [ICDAR-2015](#ICDAR-2015)
  * [IIIT](#IIIT)
  * [FFHQ](#FFHQ)
- [Material](#Material)
  * [Books](#Books)
  * [Papers](#Papers)
  * [Blogs](#Blogs)

## Frameworks

### Image processing libraries

| Framework                                                    | Image | Video | C/C++ | Python |
| ------------------------------------------------------------ | ----- | ----- | ----- | ------ |
| [<img src="https://opencv.org/wp-content/uploads/2020/07/cropped-Fav-32x32.png" width=32/>OpenCV](https://opencv.org/) | :heavy_check_mark:     | :heavy_check_mark:     | :heavy_check_mark:     | :heavy_check_mark:      |
| [<img src="https://pillow.readthedocs.io/en/stable/_static/favicon.ico" width=32 />Pillow](https://pillow.readthedocs.io/en/stable/) | :heavy_check_mark:     |       | :heavy_check_mark:     |        |
| [<img src="https://scikit-image.org/_static/favicon.ico" width=32/>Scikit-image](https://scikit-image.org/) | :heavy_check_mark:     |       | :heavy_check_mark:     |        |

### Image Augmentation

| Library                                                      | Basic Transforms | Keypoints | Bounding Boxes | Segmentation |
| ------------------------------------------------------------ | ---------------- | --------- | -------------- | ------------ |
| [Torchvision](https://pytorch.org/vision/stable/transforms.html) | :heavy_check_mark:                |           |                |              |
| [imgaug](https://imgaug.readthedocs.io/en/latest/index.html) | :heavy_check_mark:                | :heavy_check_mark:         | :heavy_check_mark:              | :heavy_check_mark:            |
| [albumentations](https://albumentations.ai/docs/)            | :heavy_check_mark:                | :heavy_check_mark:         | :heavy_check_mark:              | :heavy_check_mark:            |

### Low-level Deep Learning Frameworks

| Framework                                                    | Creator       | Python | C/C++ | R    | Java | Javascript |
| ------------------------------------------------------------ | ------------- | ------ | ----- | ---- | ---- | ---------- |
| [<img src="https://pytorch.org/favicon.ico" width="32" />PyTorch](#-Pytorch) | Facebook      | :heavy_check_mark:      | :heavy_check_mark:     |      |      |            |
| [<img src="https://www.tensorflow.org/favicon.ico" width="32" />Tensorflow](#-TensorFlow) | Google        | :heavy_check_mark:      | :heavy_check_mark:     |      | :heavy_check_mark:    | :heavy_check_mark:          |
| [<img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/>Caffe](#-Caffe) | UC Berkeley   | :heavy_check_mark:      | :heavy_check_mark:     |      |      |            |
| [<img src="https://pjreddie.com/static/icon.png" width="32"/>Darknet](#-Darknet) | Joseph Redmon |        | :heavy_check_mark:     |      |      |            |
| [<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png" width="32"/>MXNet](#-MXNet) | Apache        | :heavy_check_mark:      | :heavy_check_mark:     | :heavy_check_mark:    | :heavy_check_mark:    |            |


#### <img src="https://pytorch.org/favicon.ico" width="32" /> Pytorch

> PyTorch is a Python package that provides
    * Tensor computation (like NumPy) with strong GPU acceleration
    * Deep neural networks built on a tape-based autograd system

* :house: **Main page**: https://pytorch.org/
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/pytorch/pytorch ![GitHub stars](https://img.shields.io/github/stars/pytorch/pytorch?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/torch ![PyPi version](https://badgen.net/pypi/v/torch/)]
* :blue_book: **Docs**: https://pytorch.org/docs/stable/index.html
*  **Docker**:
    * [DockerHub](https://hub.docker.com/r/pytorch/pytorch) ![docker pulls](https://img.shields.io/docker/pulls/pytorch/pytorch.svg?style=social)
    * [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

#### <img src="https://www.tensorflow.org/favicon.ico" width="32" /> TensorFlow 

> TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

* :house: **Main page**: https://www.tensorflow.org/?hl=es-419
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/tensorflow/tensorflow  ![GitHub stars](https://img.shields.io/github/stars/tensorflow/tensorflow?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/tensorflow [![PyPi version](https://badgen.net/pypi/v/tensorflow/)](https://pypi.org/project/tensorflow)
* :blue_book: **Docs**: https://www.tensorflow.org/api_docs
* **Docker**:
    * [DockerHub](https://hub.docker.com/r/tensorflow/tensorflow) ![docker pulls](https://img.shields.io/docker/pulls/tensorflow/tensorflow.svg?style=social)
    * [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)
    
#### <img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/> Caffe

> Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license.

* :house: **Main page**: https://caffe.berkeleyvision.org/
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/BVLC/caffe ![GitHub stars](https://img.shields.io/github/stars/BVLC/caffe?style=social)
* :blue_book: **Docs**: https://caffe.berkeleyvision.org/tutorial/
* **Docker**:
    * [DockerHub](https://hub.docker.com/r/bvlc/caffe) ![docker pulls](https://img.shields.io/docker/pulls/bvlc/caffe.svg?style=social)
    * [Nnidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/caffe)
    
#### <img src="https://pjreddie.com/static/icon.png" width="32"/> Darknet

> Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

* :house: **Main page**: https://pjreddie.com/darknet/
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**:
    * https://github.com/pjreddie/darknet ![GitHub stars](https://img.shields.io/github/stars/pjreddie/darknet?style=social)
    * https://github.com/AlexeyAB/darknet ![GitHub stars](https://img.shields.io/github/stars/AlexeyAB/darknet?style=social)
* :blue_book: **Docs**: https://pjreddie.com/darknet/install/
* **Docker**:
    * [DockerHub](https://hub.docker.com/r/daisukekobayashi/darknet) ![docker pulls](https://img.shields.io/docker/pulls/daisukekobayashi/darknet.svg?style=social)

#### <img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png" width="32"/> MXNet

> A truly open source deep learning framework suited for flexible research prototyping and production.

* :house: **Main page**: https://mxnet.apache.org/versions/1.9.0/
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/apache/incubator-mxnet ![GitHub stars](https://img.shields.io/github/stars/apache/incubator-mxnet?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/mxnet [![PyPi version](https://badgen.net/pypi/v/mxnet/)](https://pypi.org/project/mxnet)
* :blue_book: **Docs**: https://mxnet.apache.org/versions/1.9.0/api

* **Docker**:
    * [DockerHub](https://hub.docker.com/r/mxnet/python) ![docker pulls](https://img.shields.io/docker/pulls/mxnet/python.svg?style=social)
    * [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/mxnet)

### High-level frameworks

| Framework                                                    | Creator   | <img src="https://pytorch.org/favicon.ico" width="32" /> | <img src="https://www.tensorflow.org/favicon.ico" width="32" /> | <img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/> | <img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png" width="32"/> |
| ------------------------------------------------------------ | --------- | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [<img src="https://pytorch.org/favicon.ico" width="32" />Torchvision](#-Torchvision) | Facebook   | :heavy_check_mark:                                                        |                                                              |                                                              |                                                              |
| [<img src="https://assets.website-files.com/5f76c986da6f6011315a6c45/5f7b485594e5335f56984a3c_Lightning_FavIcon.png" width="32">PyTorch Lightning](#-PyTorch-Lightning) | CILVR     | :heavy_check_mark:                                                        |                                                              |                                                              |                                                              |
| [<img src="https://keras.io/favicon.ico" width="32">Keras](#-Keras) | Google    |                                                          | :heavy_check_mark:                                                            |                                                              |                                                              |
| [<img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/>GluonCV](#-GluonCV) | DMLC      | :heavy_check_mark:                                                        |                                                              |                                                              | :heavy_check_mark:                                                            |
| [<img src="https://mediapipe.dev/assets/img/favicon.svg" width="32">MediaPipe](#-MediaPipe) | Google    |                                                          |                                                              |                                                              |                                                              |
| [<img src="https://production-media.paperswithcode.com/libraries/dete.png" width="32">Detectron2](#-Detectron2) | Facebook  | :heavy_check_mark:                                                        |                                                              |                                                              |                                                              |
| [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>MMCV](#-MMCV) | OpenMMLab | :heavy_check_mark:                                                        |                                                              |                                                              |                                                              |

#### <img src="https://pytorch.org/favicon.ico" width="32" /> Torchvision

- :house: **Main page**: https://pytorch.org/vision/stable/index.html
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/pytorch/vision ![GitHub stars](https://img.shields.io/github/stars/pytorch/vision?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/torchvision [![PyPi version](https://badgen.net/pypi/v/torchvision/)](https://pypi.org/project/torchvision)
- :blue_book: **Docs**: https://pytorch.org/vision/stable/index.html
    - [Image classification](https://pytorch.org/vision/stable/models.html#classification)
    - [Object detection](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
    - [Semantic segmentation](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
- :floppy_disk: **Datasets**: https://pytorch.org/vision/stable/datasets.html
- :monkey: **Model zoo**: https://pytorch.org/vision/stable/models.html

#### <img src="https://assets.website-files.com/5f76c986da6f6011315a6c45/5f7b485594e5335f56984a3c_Lightning_FavIcon.png" width="32"> PyTorch Lightning

- :house: **Main page**: https://www.pytorchlightning.ai/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/PyTorchLightning/pytorch-lightning ![GitHub stars](https://img.shields.io/github/stars/PyTorchLightning/pytorch-lightning?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/pytorch-lightning [![PyPi version](https://badgen.net/pypi/v/pytorch-lightning/)](https://pypi.org/project/pytorch-lightning)
- :blue_book: **Docs**: https://pytorch-lightning.readthedocs.io/en/stable/
    - [Image classification](https://www.analyticsvidhya.com/blog/2021/06/image-recognition-using-pytorch-lightning/)

#### <img src="https://keras.io/favicon.ico" width="32"> Keras

- :house: **Main page**: https://keras.io/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/keras-team/keras ![GitHub stars](https://img.shields.io/github/stars/keras-team/keras?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/keras [![PyPi version](https://badgen.net/pypi/v/keras/)](https://pypi.org/project/keras)
- :blue_book: **Docs**: https://keras.io/api/
    - [Image classification](https://keras.io/examples/vision/image_classification_from_scratch/)
    - [Object detection](https://keras.io/examples/vision/retinanet/)
    - [Semantic Segmentation](https://keras.io/examples/vision/oxford_pets_image_segmentation/)
- :floppy_disk: **Datasets**: https://keras.io/api/datasets/
- :monkey: **Model zoo**: https://keras.io/api/applications/

#### <img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/> GluonCV

- :house: **Main page**: https://cv.gluon.ai/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/dmlc/gluon-cv ![GitHub stars](https://img.shields.io/github/stars/dmlc/gluon-cv?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/gluoncv [![PyPi version](https://badgen.net/pypi/v/gluoncv/)](https://pypi.org/project/gluoncv)
- :blue_book: **Docs**: https://cv.gluon.ai/install.html
    - [Image classification](https://cv.gluon.ai/model_zoo/classification.html)
    - [Object Detection](https://cv.gluon.ai/model_zoo/detection.html)
    - [Segmentation](https://cv.gluon.ai/model_zoo/segmentation.html)
    - [Pose estimation](https://cv.gluon.ai/model_zoo/pose.html)
    - [Action recognition](https://cv.gluon.ai/model_zoo/action_recognition.html)
    - [Depth estimation](https://cv.gluon.ai/model_zoo/depth.html)
- :floppy_disk: **Datasets**: https://cv.gluon.ai/api/data.datasets.html
- :monkey: **Model zoo**: https://cv.gluon.ai/model_zoo/index.html

#### <img src="https://mediapipe.dev/assets/img/favicon.svg" width="32"> MediaPipe

- :house: **Main page**: https://mediapipe.dev/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/google/mediapipe ![GitHub stars](https://img.shields.io/github/stars/google/mediapipe?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/mediapipe [![PyPi version](https://badgen.net/pypi/v/mediapipe/)](https://pypi.org/project/mediapipe)
- :blue_book: **Docs**: https://google.github.io/mediapipe/getting_started/python
    - [Face detection](https://google.github.io/mediapipe/solutions/face_detection)
    - [Face mesh](https://google.github.io/mediapipe/solutions/face_mesh)
    - [Iris detection](https://google.github.io/mediapipe/solutions/iris)
    - [Hands tracking](https://google.github.io/mediapipe/solutions/hands.html)
    - [Pose detection](https://google.github.io/mediapipe/solutions/pose)
    - [Selfie segmentation](https://google.github.io/mediapipe/solutions/selfie_segmentation)
- :monkey: **Model zoo**: https://google.github.io/mediapipe/solutions/solutions

#### <img src="https://production-media.paperswithcode.com/libraries/dete.png" width="32"> Detectron2

- :house: **Main page**: https://github.com/facebookresearch/detectron
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/facebookresearch/detectron2 ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/detectron2?style=social)
- :blue_book: **Docs**: https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html
    - [Object detection](https://www.analyticsvidhya.com/blog/2021/08/your-guide-to-object-detection-with-detectron2-in-pytorch/)
- :floppy_disk: **Datasets**: https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html
- :monkey: **Model zoo**: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
  
#### <img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/> MMCV

- :house: **Main page**: https://openmmlab.com/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/open-mmlab/mmcv ![GitHub stars](https://img.shields.io/github/stars/open-mmlab/mmcv?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:
	- **MMCV**: https://pypi.org/project/mmcv [![PyPi version](https://badgen.net/pypi/v/mmcv/)](https://pypi.org/project/mmcv)
	- **MMClassification**: https://pypi.org/project/mmcls [![PyPi version](https://badgen.net/pypi/v/mmcls)](https://pypi.org/project/mmcls)
	- **MMDetection**: https://pypi.org/project/mmdet [![PyPi version](https://badgen.net/pypi/v/mmdet/)](https://pypi.org/project/mmdet)
	- **MMSegmentation**: https://pypi.org/project/mmsegmentation [![PyPi version](https://badgen.net/pypi/v/mmsegmentation/)](https://pypi.org/project/mmsegmentation)
	- **MMPose**: https://pypi.org/project/mmpose [![PyPi version](https://badgen.net/pypi/v/mmpose/)](https://pypi.org/project/mmpose)
- :blue_book: **Docs**: https://mmcv.readthedocs.io/en/latest/
    - [Image classification (MMClassification)](https://mmclassification.readthedocs.io/en/latest/)
    - [Object detection (MMDetection)](https://mmdetection.readthedocs.io/en/latest/)
    - [Image segmentation (MMSegmentation)](https://mmsegmentation.readthedocs.io/en/latest/)
    - [Pose detection (MMPose)](https://mmpose.readthedocs.io/en/latest/)
    - [Action detection (MMAction2)](https://mmaction2.readthedocs.io/en/latest/)

## Tasks

### :camera: Image classification

Image classification refers to the task of extracting information classes from a multiband raster image.

<img src="https://data-flair.training/blogs/wp-content/uploads/sites/2/2020/05/Cats-Dogs-Classification-deep-learning.gif" width="50%" />

#### Models

| Model                                   | Paper                                                        | Published |
| :-------------------------------------- | :----------------------------------------------------------- | :-------- |
| <a name="AlexNet">AlexNet</a>           | [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997) | Apr-2014  |
| <a name="GoogleLeNet">GoogLeNet</a>     | [Going Deeper with Convolutions](arxiv.org/abs/1409.4842)    | Sep-2014  |
| <a name="VGG">VGG</a>                   | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | Sep-2014  |
| <a name="InceptionV3">InceptionV3</a>   | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | Dec-2015  |
| <a name="ResNet">ResNet</a>             | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | Dec-2015  |
| <a name="SqueezeNet">SqueezeNet</a>     | [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360) | Feb-2016  |
| <a name="Wide ResNet">Wide ResNet</a>   | [Wide Residual Networks](https://arxiv.org/abs/1605.07146)   | May-2016  |
| <a name="DenseNet">DenseNet</a>         | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | Aug-2016  |
| <a name="ResNeXt">ResNeXt</a>           | [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) | Nov-2016  |
| <a name="DarknetV1">DarknetV1</a>       | [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242v1) | Dec-2016  |
| <a name="MobileNetV2">MobileNetV2</a>   | [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) | Jan-2018  |
| <a name="DarknetV2">DarknetV2</a>       | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767v1) | Apr-2018  |
| <a name="MNASNet">MNASNet</a>           | [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) | Jul-2018  |
| <a name="MobileNetV3">MobileNetV3</a>   | [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) | May-2019  |
| <a name="EfficientNet">EfficientNet</a> | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | May-2019  |
| <a name="ViT">ViT</a>                   | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | Oct-2020  |
| <a name="Swim">Swin</a>                 | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | Mar-2021  |

#### Pretrained models

* [<img src="https://pytorch.org/favicon.ico" width="32" />Torchvision](https://pytorch.org/vision/stable/models.html#classification)
* [<img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/>GluonCV](https://cv.gluon.ai/model_zoo/classification.html)
* [<img src="https://keras.io/favicon.ico" width="32">Keras](https://keras.io/api/applications/#available-models)
* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMClassification)](https://mmclassification.readthedocs.io/en/latest/model_zoo.html)

#### Benchmark

| Family       | Network            | Year | #P (*M*) | *  Acc@1 |
| ------------ | ------------------ | ---- | -------- | -------- |
| [AlexNet](#AlexNet)      | AlexNet            | Apr-2014 | 61       | 56.522   |
| [VGG](#VGG)          | VGG-11 (BN)        | Sep-2014 | 133      | 70.370   |
| [VGG](#VGG)          | VGG-13 (BN)        | Sep-2014 | 133      | 71.586   |
| [VGG](#VGG)          | VGG-16 (BN)        | Sep-2014 | 138      | 73.360   |
| [VGG](#VGG)          | VGG-19 (BN)        | Sep-2014 | 144      | 74.218   |
| [GoogLeNet](#GoogLeNet)    | GoogLeNet          | 2014 | 13       | 69.778   |
| [ResNet](#ResNet)       | ResNet-18          | 2015 | 12       | 69.758   |
| [ResNet](#ResNet)       | ResNet-34          | 2015 | 22       | 73.314   |
| [ResNet](#ResNet)       | ResNet-50          | 2015 | 26       | 76.130   |
| [ResNet](#ResNet)       | ResNet-50          | 2015 | 45       | 77.374   |
| [ResNet](#ResNet)       | ResNet-101         | 2015 | 50       | 78.312   |
| [InceptionV3](#InceptionV3) | Inception-V3       | 2015 | 27       | 77.294   |
| [SqueezeNet](#SqueezeNet)   | SqueezeNet 1.0     | 2016 | 1        | 58.092   |
| [SqueezeNet](#SqueezeNet)   | SqueezeNet 1.1     | 2016 | 1        | 58.178   |
| [DenseNet](#DenseNet)     | DenseNet-121       | 2016 | 8        | 74.434   |
| [DenseNet](#DenseNet)     | DenseNet-126       | 2016 | 29       | 75.600   |
| [DenseNet](#DenseNet)     | DenseNet-169       | 2016 | 14       | 75.600   |
| [DenseNet](#DenseNet)     | DenseNet-201       | 2016 | 20       | 76.896   |
| [DarknetV1](#DarknetV1) | Darknet-19         | 2016 | ?        | 72.9     |
| [Darknet](#Darknet)      | Darknet-53         | 2016 | ?        | 77.2     |
| [Wide ResNet](#Wide-ResNet)  | Wide ResNet-50-2   | 2017 | 69       | 78.468   |
| [Wide ResNet](#Wide-ResNet)   | Wide ResNet-101-2  | 2017 | 127      | 78.848   |
| [MobileNet](#MobileNet)    | MobileNet-v2       | 2018 | 4        | 71.878   |
| [MobileNet](#MobileNet)    | MobileNet-v3-small | 2018 | 3        | 67.668   |
| [MobileNet](#MobileNet)    | MobileNet-v3-large | 2018 | 5        | 74.042   |
| [MNASNet](#MNASNet)      | MNASNet 0-5        | 2018 | 2        | 67.734   |
| [MNASNet](#MNASNet)      | MNASNet 0-75       | 2018 | 3        | ??       |
| [MNASNet](#MNASNet)      | MNASNet 1-0        | 2018 | 4        | 73.456   |
| [MNASNet](#MNASNet)      | MNASNet 1-3        | 2018 | 6        | ??       |
| [EfficientNet](#EfficientNet) | EfficientNet B0    | 2019 | 5        | 77.692   |
| [EfficientNet](#EfficientNet) | EfficientNet B1    | 2019 | 8        | 78.642   |
| [EfficientNet](#EfficientNet) | EfficientNet B2    | 2019 | 9        | 80.608   |
| [EfficientNet](#EfficientNet) | EfficientNet B3    | 2019 | 12       | 82.008   |
| [EfficientNet](#EfficientNet) | EfficientNet B4    | 2019 | 19       | 83.384   |
| [EfficientNet](#EfficientNet) | EfficientNet B5    | 2019 | 30       | 83.444   |
| [EfficientNet](#EfficientNet) | EfficientNet B6    | 2019 | 43       | 84.008   |
| [EfficientNet](#EfficientNet) | EfficientNet B6    | 2019 | 66       | 84.122   |
| [Swin](#Swin)         | Swin-S             | 2021 | 49       | 83.21    |
| [Swin](#Swin)         | Swin-B             | 2021 | 87       | 85.16    |
| [Swin](#Swin)         | Swin-L             | 2021 | 196      | 86.24    |
| [ViT](#ViT)          | ViT-L/14           | 2020 | 307      | 87.46    |
| [ViT](#ViT)          | ViT-H/14           | 2020 | 632      | 88.55    |



* Acc@1: Top-1 Accuracy on ImageNet at 224x224 resolution

### :mag: Object detection

>  **Object detection** refers to identifying the **location** of one or more objects in an image with its **bounding box**.

<img src="https://miro.medium.com/max/1000/1*NLnnf_M4Nlm4p1GAWrWUCQ.gif" width="50%">

#### Models

| Model | Paper | Published |
|----------|---------|---------------|
| <a name="R-CNN">R-CNN</a> | [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) | Oct-2013 |
| <a name="FastR-CNN">Fast R-CNN</a> | [Fast R-CNN](https://arxiv.org/abs/1504.08083) | Sep-2015 |
| <a name="FasterR-CNN">Faster R-CNN</a> | [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) | Jan-2016 |
| <a name="YOLOv1">YOLOv1</a> | [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) | Jun-2015 |
| <a name="YOLOv2">YOLOv2</a> | [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242v1) | Dec-2016 |
| <a name="SSD">SSD</a> | [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) | Dec-2015 |
| <a name="YOLOv3">YOLOv3</a> | [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767v1) | Apr-2018 |
| <a name="RetinaNet">RetinaNet</a> | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) | Feb-2018 |
| <a name="MaskRCNN">MaskRCNN</a> | [Mask R-CNN](https://arxiv.org/abs/1703.06870) | Mar-2017 |
| <a name="CascadeR-CNN">Cascade R-CNN</a> | [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726) | Dec-2017 |
| <a name="DETR">DETR</a> | [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) | May-2020 |

#### Pretrained models

* [<img src="https://pytorch.org/favicon.ico" width="32" />Torchvision](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
* [<img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/>GluonCV](https://cv.gluon.ai/model_zoo/detection.html)
* [<img src="https://keras.io/favicon.ico" width="32">Keras](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMDetection)](https://mmdetection.readthedocs.io/en/v2.20.0/model_zoo.html)
* [<img src="https://production-media.paperswithcode.com/libraries/dete.png" width="32">Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-object-detection-baselines)

#### Benchmark

| Family       | Network                  | Backbone          | Year | * AP |
|----|----|----|---|--|
| [R-CNN](#R-CNN)        | R-CNN                    | ??                | 2013 | -    |
| [Fast R-CNN](#FastR-CNN)   | Fast R-CNN               | [VGG16](#VGG)  | 2015 | 19.7 |
| [Faster R-CNN](#FasterR-CNN) | Faster R-CNN             | [VGG16](#VGG)                 | 2016 | 21.9 |
| [YOLOv1](#YOLOv1)     | YOLO V1                  | * [GoogLeNet](#GoogLeNet) | 2015 | -    |
| [YOLOv2](#YOLOv2)     | YOLO V2                  | [Darknet-19](DarknetV1) | 2016 | 21.6 |
| [SSD](#SSD)          | SSD300                   | [VGG16](#VGG)                 | 2016 | 23.2 |
| [SSD](#SSD)          | SSD500                   | [VGG16](#VGG)                 | 2016 | 26.8 |
| [Cascade R-CNN](#CascadeR-CNN) | Cascade-R-CNN-100        | [ResNet-101](#ResNet) | 2017 | 42.8 |
| [YOLOv3](#YOLOv3)     | YOLO V3                  | [Darknet-53](#DarknetV2) | 2018 | 33.0 |
| [RetinaNet](#RetinaNet)    | RetinaNet-ResNet-101     | [ResNet-101-FPN](#ResNet) | 2017 | 39.1 |
| [RetinaNet](#RetinaNet)    | RetinaNet-ResNeXt-101    | [ResNeXt-101-FPN](#ResNeXt) | 2017 | 40.8 |
| [MaskRCNN](#MaskRCNN)     | MaskRCNN R-101-FPN       | [ResNet-101-FP](#ResNet) | 2018 | 40.8 |
| [MaskRCNN](#MaskRCNN)     | MaskRCNN X-101-64x4d-FPN | [ResNeXt-101-64x4d](#ResNeXt) | 2018 | 42.7 |
| [DETR](#DETR)         | DETR-DC5                 | [ResNet-50 + DC](#ResNet) | 2020 | 43.3 |
| [DETR](#DETR)         | DETR-DC5-R101            | [ResNet-101 + DC](#ResNet) | 2020 | 44.9 |

* AP: AP[.5:.05:0.95] on COCO test-dev

### :busts_in_silhouette: Image Segmentation

>  **Image Segmentation** is the process of partitioning a [digital image](https://en.wikipedia.org/wiki/Digital_image) into multiple **image segments**, also known as **image regions** or **image objects**

<img src="https://miro.medium.com/max/1000/1*RZnBSB3QpkIwFUTRFaWDYg.gif" width="50%"/>



#### Models

| Model | Paper | Published |
|----------|----------|--------------|
| <a name="UNet">U-Net</a> | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | May-2015 |
| <a name="FCN-old">FCN</a> | [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) | Nov-2014 |
| <a name="FCN">FCN</a> | [Improving Fully Convolution Network for Semantic Segmentation](https://arxiv.org/pdf/1611.08986.pdf) | Nov-2016 |
| <a name="DeepLabV3">DeepLabV3</a> | [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3) | Jun-2017 |
| <a name="PSPNet">PSPNet</a> | [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105) | Dec-2016 |
| <a name="UPerNet">UPerNet</a> | [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) | Jul-2018 |
| <a name="DANet">DANet</a> | [Dual Attention Network for Scene Segmentation](https://arxiv.org/abs/1809.02983) | Sep-2018 |

#### Pretrained models

* [<img src="https://pytorch.org/favicon.ico" width="32" />Torchvision](https://pytorch.org/vision/stable/models.html#semantic-segmentation)
* [<img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/>GluonCV](https://cv.gluon.ai/model_zoo/segmentation.html)
* [<img src="https://keras.io/favicon.ico" width="32">Keras](https://github.com/qubvel/segmentation_models#models-and-backbones)
* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMSegmentation)](https://mmsegmentation.readthedocs.io/en/latest/model_zoo.html)
* [<img src="https://production-media.paperswithcode.com/libraries/dete.png" width="32">Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-object-detection-baselines)


#### Benchmark

| Family    | Backbone                 | Year | * mIoU |
| --------- | ------------------------ | ---- | ------ |
| UNet      | UNet-S5-D16              | 2016 | 69.10  |
| FCN       | [ResNet-18-D8](#ResNet)  | 2017 | 71.11  |
| FCN       | [ResNet-50-D8](#ResNet)  | 2017 | 73.61  |
| FCN       | [ResNet-101-D8](#ResNet) | 2017 | 76.80  |
| DeepLabV3 | [ResNet-50-D8](#ResNet)  | 2017 | 77.85  |
| PSPNet    | [ResNet-101-D8](#ResNet) | 2016 | 78.34  |
| UPerNet   | [ResNet-50](#ResNet)     | 2018 | 78.19  |
| UPerNet   | [ResNet-101](#ResNet)    | 2018 | 79.40  |
| DANet     | [ResNet-50-D8](#ResNet)  | 2018 | 79.34  |
| DANet     | [ResNet-101-D8](#ResNet) | 2018 | 80.41  |

* mIoU: Mean IoU on CityScapes at 512x1024 resolution

### :pushpin: Landmark/Keypoint Extraction

> Landmark/Keypoint Extraction is the process of determining spatial key-points of an object in an image (e.g: Pose keypoints)

<img src="https://1.bp.blogspot.com/-bLvChKpaja8/XcCkcSDcGoI/AAAAAAAAAw0/Oc2vMBq0IkQ5tljHc24eiv8Zu0qz-9pBwCLcBGAsYHQ/s320/p2.gif" width="50%"/>

#### Methods

1. Cascade
2. Heatmap methods
	1. Top-down heatmap
	2. Bottom-up heatmap
	3. Multi-Scale High-Resolution Networks

#### Pretrained models

##### Pose detection

* [<img src="https://pytorch.org/favicon.ico" width="32" />Torchvision](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
* [<img src="https://cv.gluon.ai/_static/assets/img/gluon_white.png" width="32"/>GluonCV](https://cv.gluon.ai/model_zoo/pose.html)
* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMPose)](https://mmpose.readthedocs.io/en/v0.22.0/modelzoo.html)
* [<img src="https://production-media.paperswithcode.com/libraries/dete.png" width="32">Detectron2](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)

Papers:

* :link: https://arxiv.org/pdf/2007.08090.pdf

#### Models

| Model                                   | Paper                                                        | Published |
| :-------------------------------------- | :----------------------------------------------------------- | :-------- |
| <a name="Deep-Pose">Deep Pose</a> | [DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/abs/1312.4659) | Dec-2013 |
| <a name="CPM">CPM</a> | [Convolutional Pose Machines](https://arxiv.org/abs/1602.00134) | Jan-2016 |
| <a name="RSN">RSN</a> | [Learning Delicate Local Representations for Multi-Person Pose Estimation](https://arxiv.org/abs/2003.04030) | Mar-2020 |
| <a name="HRNet">HRNet</a> | [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/abs/1908.07919) | Aug-2019 |


#### Benchmark

| Family      | Method          | Backbone       | Year | * AP |
| ----------- | --------------- | -------------- | ---- | ---- |
| [Deep Pose](#Deep-Pose) | Cascade         | [Resnet-50](#ResNet) | 2014 | 52.6 |
| [Deep Pose](#Deep-Pose) | Cascade         | [Resnet-101]((#ResNet)) | 2014 | 56.0 |
| [Deep Pose](#Deep-Pose) | Cascade         | [Resnet-152]((#ResNet)) | 2014 | 58.3 |
| [CPM](#CPM) | Top-down heatmap | ?              | 2016 | 62.3 |
| ResnetV1    | Top-down Heatmap | [ResnetV1D-50]((#ResNet)) | 2019 | 72.2 |
| ResnetV1    | Top-down Heatmap | [ResnetV1D-100]((#ResNet)) | 2019 | 73.1 |
| ResnetV1    | Top-down Heatmap | [Resnet-152](#ResNet) | 2019 | 73.7 |
| VGG         | Top-down Heatmap | [VGG-16](#VGG) | 2015 | 69.8 |
| Mobilenetv2 | Top-down Heatmap | [MobileNetV2](#MobileNetV2) | 2018 | 64.6 |
| [RSN](#RSN) | Top-down Heatmap | ResNet-18    | 2020 | 70.4 |
| [RSN](#RSN)             | Top-down Heatmap | 3x ResNet-50 | 2020 | 75.0 |
| [HRNet](HRNet) | Multi-Scale High-Resolution Networks | HRNet-w48 | 2019 | 75.6 |

* AP: Average precision on COCO-2017 at 256x192 resolution

### :triangular_ruler: Metric Learning / Few-Shot Learning

#### Methods

1. Siamese Networks
2. Meta-Learning

#### Models

| Paper | Backbone | R@1 | Published |
|---------|------------|---------|---------|
| [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) | _Custom_                |      | Aug-2016 |
| [Hardness-Aware Deep Metric Learning](https://arxiv.org/abs/1903.05503)  | [GoogLeNet](#GoogLeNet) | 43.6 | Mar-2019 |
| [Local Similarity-Aware Deep Feature Embedding](https://arxiv.org/abs/1610.08904) | [GoogLeNet](#GoogLeNet) | 58.3 | Oct-2016 |
| [Hard-Aware Deeply Cascaded Embedding](https://arxiv.org/abs/1611.05720) | [GoogLeNet](#GoogLeNet) | 60.7 | Nov-2016 |
| [Sampling Matters in Deep Embedding Learning](https://arxiv.org/abs/1706.07567) | [ResNet-50](#ResNet)    | 63.9 | Jun-2017 |
| [SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](https://arxiv.org/abs/1909.05235) | [GoogLeNet](#GoogLeNet) | 65.4 | Sep-2019 |
| [Calibrated neighborhood aware confidence measure for deep metric learning](https://arxiv.org/abs/2006.04935v1) | ?? | 74.9 | Jun-2020 |
| [A Closer Look at Few-shot Classification](https://arxiv.org/abs/1904.04232) | Conv4 | 60.5 | Jan-2020 |
| [Negative Margin Matters: Understanding Margin in Few-shot Classification](https://arxiv.org/abs/2003.12060) | [ResNet-18](#ResNet) | 72.7 | Mar-2020 |
| [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf) | [GoogLeNet](#GoogLeNet) | 54.6 | Jun-2016 |

#### Pretrained models

* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMFewShot)](https://mmfewshot.readthedocs.io/en/latest/model_zoo.html)

### :bookmark_tabs: OCR

#### Text Detection

##### Models

| Paper | Published | * R | * P | * F |
|-------|-----------|--------|-----------|----------|
| [Multi-Oriented Text Detection with Fully Convolutional Networks](https://arxiv.org/abs/1604.04018v2) | Apr-2016 | 88 | 78 | 73 |
| [Single Shot Text Detector with Regional Attention](https://arxiv.org/abs/1709.00138v1) | Sep-2017 | 86 | 88 | 87 |
| [Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network](https://arxiv.org/abs/1908.05900v2) | Aug-2019 | 81 | 84 | 82 |
| [TextFuseNet: Scene Text Detection with Richer Fused Features](https://www.ijcai.org/proceedings/2020/72) | Jul-2020 | 89 | 91 | 90 |

* R, P, F: Recall, Precision and F1-Score on ICDAR-2015

##### Pretrained models

* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMOCR)](https://mmocr.readthedocs.io/en/latest/modelzoo.html#text-detection-models)

#### Text recognition

##### Models

| Paper | Published | * Acc |
|-------|-----------|--------|
| [Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://arxiv.org/abs/2103.06495) | Mar-2021 | 96.2 |

* Acc: Accuracy on IIIT dataset

##### Pretrained models

* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMOCR)](https://mmocr.readthedocs.io/en/latest/modelzoo.html#text-recognition-models)

#### Solutions

- Standalone:
	1. Tesseract: https://github.com/tesseract-ocr/tesseract
	2. PyTesseract: https://github.com/madmaze/pytesseract

- Cloud:
	1. Google Cloud: https://cloud.google.com/vision/docs/ocr
	2. Microsoft Azure: https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/client-library?tabs=visual-studio&pivots=programming-language-python
	3. ABBYY: https://www.abbyy.com/cloud-ocr-sdk/legal/

### :art: Image Synthesis

> Image synthesis is the process of artificially generating images that contain some particular desired content

#### Unconditional

<img src="https://miro.medium.com/max/1280/1*9p3vCVmRHY5QoPQM98Kx0A.gif" width="50%" />

##### Models

| Model | Paper | Published |
|-------|-------|-----------|
| GAN | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) | Jun-2014 |
| DCGAN | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) | Nov-2015 |
| VAE-GAN | [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300) | Dec-2015 |
| InfoGAN | [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657v1) | Jan-2016 |
| BiGAN | [Adversarial Feature Learning](https://arxiv.org/abs/1605.09782v7) | May-2016 |
| LSGAN | [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076) | Nov-2016 |
| WGAN | [Wasserstein GAN](https://arxiv.org/abs/1701.07875v3) | Jan-2017 |
| GGAN | [Geometric GAN](https://arxiv.org/abs/1705.02894) | May-2017 |
| ProGAN | [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) | Oct-2017 |
| StyleGAN | [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) | Dec-2018 |
| LOGAN | [LOGAN: Latent Optimisation for Generative Adversarial Networks](https://arxiv.org/abs/1912.00953) | Dec-2019 |
| StyleGAN2 | [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) | Dec-2019 |
| Improved-DDPM | [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) | Feb-2021 |
| StyleGAN3 | [Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) | Jun-2021 |
| SwinStyle | [StyleSwin: Transformer-based GAN for High-resolution Image Generation](https://arxiv.org/abs/2112.10762) | Dec-2021 |

#### Class-Conditioned

<img src="https://learnopencv.com/wp-content/uploads/2021/07/Conditional-GAN-in-PyTorch-and-TensorFlow.jpeg" width="50%" />

##### Models

| Model | Paper | Published |
|-------|-------|-----------|
| CGAN | [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) | Nov-2014 |
| SNGAN | [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957) | Feb-2018 |
| SAGAN | [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318) | May-2018 |
| BigGAN | [Large Scale Adversarial Representation Learning](https://arxiv.org/abs/1907.02544v2) | Jul-2019 |

#### Image to Image Translation

<img src="https://raw.githubusercontent.com/gradpratik/CycleGAN-Tensorflow-2/master/pics/horse2zebra.gif" width="50%" />

##### Models

| Model | Paper | Published |
|-------|-------|-----------|
| CoGAN | [Coupled Generative Adversarial Networks](https://arxiv.org/abs/1606.07536) | Jun-2016 |
| Pix2Pix | [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) | Nov-2016 |
| CycleGAN | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) | Mar-2017 |
| CC-FPSE-AUG | [Improving Augmentation and Evaluation Schemes for Semantic Image Synthesis](https://arxiv.org/abs/2011.12636) | Nov-2020 |
| DAFormer | [DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation](https://arxiv.org/abs/2111.14887) | Nov-2021 |

#### Neural Style-Transfer

##### Models

| Paper | Published |
|-------|-----------|
| [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) | Aug-2015 |
| [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022v3) | Jul-2016 |
| [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/abs/1604.04382) | Apr-2016 |
| [Universal Style Transfer via Feature Transforms](https://arxiv.org/abs/1705.08086) | May-2017 |
| [A Closed-form Solution to Photorealistic Image Stylization](https://arxiv.org/abs/1802.06474v5) | Feb-2018 |

#### Pretrained Models

* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMGeneration)](https://mmgeneration.readthedocs.io/en/latest/modelzoo_statistics.html)
* <img src="https://www.tensorflow.org/favicon.ico" width="32" /> Tensorflow:
	* [StyleGAN](https://github.com/NVlabs/stylegan#using-pre-trained-networks)
	* [StyleGAN2](https://github.com/NVlabs/stylegan2#using-pre-trained-networks)

### :paintbrush: Image Editing

#### Super-Resolution

<img src="https://miro.medium.com/max/1400/0*cUNPPG12JvHIJCcg.gif" width="50%" />

##### Models

| Model | Paper | Published |
|-------|-------|-----------|
| SR-CNN | [Image Super-Resolution Using Deep Convolutional Networks](https://arxiv.org/abs/1501.00092) | Dec-2014 |
| SRGan | [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) | Sep-2016 |
| EDSR | [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/abs/1707.02921) | Jul-2017 |
| ESRGAN | [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) | Sep-2018 |
| HAN | [Single Image Super-Resolution via a Holistic Attention Network](https://arxiv.org/abs/2008.08767v1) | Aug-2020 |
| GLEAN | [GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution](https://arxiv.org/abs/2012.00739) | Dec-2020 |
| SwinIR | [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257v1) | Aug-2021 |
| BRGM | [Bayesian Image Reconstruction using Deep Generative Models](https://arxiv.org/abs/2012.04567v5) | Dec-2020 |

#### Inpainting

<img src="https://thumbs.gfycat.com/BetterAstonishingBird-max-1mb.gif" width="50%" />

##### Models

| Model | Paper | Published |
|-------|-------|-----------|
| BRGM | [Bayesian Image Reconstruction using Deep Generative Models](https://arxiv.org/abs/2012.04567v5) | Dec-2021 |
| DeepFillV1 | [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892) | Jan-2018 |
| DeepFillV2 | [https://arxiv.org/abs/1806.03589](https://arxiv.org/abs/1806.03589) | Jun-2018 |
| SRNet | [Editing Text in the Wild](https://arxiv.org/abs/1908.03047v1) | Aug-2019 |
| GIP | [Image Completion and Extrapolation with Contextual Cycle Consistency](https://arxiv.org/abs/2006.02620) | Jun-2020 | 

### Pretrained Models

* [<img src="https://oss.openmmlab.com/www/community/openmmlab.png" width=32/>Open MMLab (MMEditing)](https://mmediting.readthedocs.io/en/latest/modelzoo.html)

## Datasets

### MNIST

> A large database of handwritten digits

* :house: **Main page**: http://yann.lecun.com/exdb/mnist/
* **#Images**: L-24x24
    * Train: 60k
    * Test: 10k
* **Classes**: (10)
    0,1,2,3,4,5,6,7,8,9
* **Tasks**:
    * Multi-class [Image Classification](###Image-classification)

### CIFAR-10

>  Labeled subsets of the 80 million tiny images dataset

* :house: **Main page**: https://www.cs.toronto.edu/~kriz/cifar.html
* **#Images**: RGB-32x32
    * Train: 50k
    * Test: 10k
* **Classes**: (10)
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
* **Tasks**:
    * Multi-class [Image classification](#camera-image-classification)
    
### CIFAR-100

> Labeled subsets of the 80 million tiny images dataset

* :house: **Main page**: https://www.cs.toronto.edu/~kriz/cifar.html
* **#Images**: RGB-32x32
    * Train: 50k
    * Test: 10k
* **Super-Classes**: (100)
    aquatic mammals, fish, flowers, food containers, fruit and vegetables,
    household electrical devices, household furniture, insects, large carnivores,
    large man-made outdoor things, large natural outdoor scenes,
    large omnivores and herbivores, medium-sized mammals, non-insect invertebrates,
    people, reptiles, small mammals, trees, vehicles 1, vehicles 2
* **Tasks**:
    * Multi-class [Image classification](#camera-image-classification)
    
### Caltech 101

> Pictures of objects belonging to 101 categories

* **Dificulty**: Mid
* :house: **Main page**: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
* **Images**: RGB ~300x200
    * Train: ?
    * Test: ?
* **Tasks**:
    * Multi-class [Image classification](#camera-image-classification)

### CelebA ###

> Large-scale face attributes dataset

* :house: **Main page**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* **Images**:
    * Train: 200k
    * Test: ??
* **Tasks**:
    * Face Detection: [Object detection](#mag-object-detection)
    * Face attributes: [Image classification](#camera-image-classification)
    * KeyPoint Extraction: [Landmark/Keypoint Extraction](#pushpin-landmarkkeypoint-extraction)
    * Identity: [Metric Learning / Few-Shot Learning](#triangular_ruler-metric-learning--few-shot-learning)
* **Classes**:
    * **Face attributes** (40):  5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, Wearing_Necktie, Young, 
    * **Landmarks**: (5)
      * Left-eye, Right-eye, Nose, Left-Mouth, Right-Mouth

### WiderFace ###

> A face detection benchmark dataset with a high degree of variability in scale, pose and occlusion

* :house: **Main page**: http://shuoyang1213.me/WIDERFACE/index.html
* **Images**:
    *  All: 32k
    *  Train: 12k
    *  Val: 3k
    *  Test: 16k
* **Tasks**:
	* Face Detection: [Object detection](#mag-object-detection)

### LFW ###

> A database of face photographs designed for studying the problem of unconstrained face recognition

* :house: **Main page**: http://vis-www.cs.umass.edu/lfw/
* **Images**: 13k
* **Tasks**:
	* Face identification: [Metric Learning / Few-Shot Learning](#triangular_ruler-metric-learning--few-shot-learning)
	* Face attributes: [Image classification](#camera-image-classification)
* **Classes**:
	* **Face attributes** (73): Male, Asian, White, Black, Baby, Child, Youth, Middle_Aged, Senior, Black_Hair, Blond_Hair, Brown_Hair, Bald, No_Eyewear, Eyeglasses, Sunglasses, Mustache, Smiling, Frowning, Chubby, Blurry, Harsh_Lighting, Flash, Soft_Lighting, Outdoor, Curly_Hair, Wavy_Hair, Straight_Hair, Receding_Hairline, Bangs, Sideburns, Fully_Visible_Forehead, Partially_Visible_Forehead, Obstructed_Forehead, Bushy_Eyebrows, Arched_Eyebrows, Narrow_Eyes, Eyes_Open, Big_Nose, Pointy_Nose, Big_Lips, Mouth_Closed, Mouth_Slightly_Open, Mouth_Wide_Open, Teeth_Not_Visible, No_Beard, Goatee, Round_Jaw, Double_Chin, Wearing_Hat, Oval_Face, Square_Face, Round_Face, Color_Photo, Posed_Photo, Attractive_Man, Attractive_Woman, Indian, Gray_Hair, Bags_Under_Eyes, Heavy_Makeup, Rosy_Cheeks, Shiny_Skin, Pale_Skin, 5_o'_Clock_Shadow, Strong_Nose-Mouth_Lines, Wearing_Lipstick, Flushed_Face, High_Cheekbones, Brown_Eyes, Wearing_Earrings, Wearing_Necktie, Wearing_Necklace


### CelebAMask-HQ ###

>  Large-scale face attributes dataset

* :house: **Main page**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* **Images**:
    * Train: 200k
    * Test: ??
* **Tasks**:
    * FaceParts segmentation: [Instance segmentation](#Image-segmentation)
    * Image segmentation: [Image editing](#paintbrush-Image-Editing)
* **Classes**: (17)
    * **Instance classes**: skin, nose, left_eye, right_eye, left_eyebrow, right_eyebrow, left_ear, right_ear, mouth, lip, hair, hat, eyeglass, earring, necklace, neck, cloth.
    
### ImageNet

> An image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images

* :house: **Main page**: https://www.image-net.org/
* **Images**:
    * Total: 14M
    * ILSVRC 2012-2017 subset:
        * **Train**: 1M
        * **Valid**: 50k
        * **Test**: 100k
* **Classes**:
    * All: (21K)
    * ILSVRC 2017: (1k)
* **Tasks**:
    * [Object detection](#mag-object-detection)
    * [Image classification](#camera-image-classification)
    
### COCO

> A large-scale object detection, segmentation, and captioning dataset.

* :house: **Main page**: https://cocodataset.org/#home
* **Images**
    * **Total**: 330K
    * **Labeled**: 200K
* **Classes**:
    * **Object Detection & Object segmentation**: 80
    * **Key-point**: 18
* **Tasks**:
    * [Object detection](#mag-object-detection)
    * Object & Stuff Segmentation:  [Image Segmentation](#busts_in_silhouette-image-segmentation)
    * Person Key-point detection: [Landmark/Keypoint Extraction](#pushpin-landmarkkeypoint-extraction)
    * [Image Captioning](#Image-Captioning)
    
### CityScapes

> A large-scale dataset that contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5000 frames in addition to a larger set of 20 000 weakly annotated frames

* :house: **Main page**: https://www.cityscapes-dataset.com/
* **Images**:
    * **Total**: 25k
    * **Val**: 
* **Clases**
    * **Flat**: road, sidewalk, parking, rail track
    * **Human**: person, rider
    * **Vehicle**:  car, truck, bus, on rails, motorcycle, bicycle, caravan, trailer
    * **Construction**: building, wall, fence, guard rail, bridge, tunnel
    * **Object**: pole, pole group, traffic sign, traffic light nature	vegetation, terrain
    * **Sky**: sky 
    * **Void**: ground, dynamic, static
* **Tasks**:
    *  [Image Segmentation](#busts_in_silhouette-image-segmentation)

### Pascal VOC

> Provides standardised image data sets for object class recognition 

* :house: **Main page**: http://host.robots.ox.ac.uk/pascal/VOC/
* **Images**:
    * **Total**: 11.5k
    * **Object Detection**: 11.5k
    * **Image Segmentation**: 6.9k
* **Classes**:
    * Person: person
    * Animal: bird, cat, cow, dog, horse, sheep
    * Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
    * Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
* **Tasks**:
    * [Object detection](#mag-object-detection)
    * [Image Segmentation](#busts_in_silhouette-image-segmentation)

### CUB-200-2011

> An extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations

* :house: **Main page**: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
* **Images**:
	* **Total**: 11k
* **Labels**: 312
* **Tasks**:
	* [Metric Learning / Few-Shot Learning](#triangular_ruler-metric-learning--few-shot-learning)
	* [Object detection](#mag-object-detection)
	* [Image classification](#camera-image-classification)

### ICDAR-2015:

* :house: **Main page**: https://iapr.org/archives/icdar2015/index.html%3Fp=254.html
* **Images**:
	* **Train**: 1k
	* **Test**: 500
* **Tasks**:
	* [Text Detection](#Text-Detection)

### IIIT:

> A dataset harvested from Google image search from Query words like billboards, signboard, house numbers, house name plates, movie posters

* 🏠 **Main page**: http://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset
* **Images**: 5k
* **Tasks**:
	* [Text recognition](#Text-recognition)

### FFHQ

> Flickr-Faces-HQ (FFHQ) is a high-quality image dataset of human faces, originally created as a benchmark for generative adversarial networks (GAN):

* :house: **Main page** : https://github.com/NVlabs/ffhq-dataset
* **Images**: 70k at 1024x1024
* **Tasks**:
	* [Image synthesis](#art-Image-synthesis)
    
## Material

### :books: Books

* [Deep Learning book](https://www.deeplearningbook.org/)
* [Computer Vision:  Models, Learning, and Inference](http://www.computervisionmodels.com/)
* [Computer Vision with Python](https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/)

### :page_facing_up: Papers

* [Arxiv](https://arxiv.org/)
* [PapersWithCode](https://paperswithcode.com/)
* [Research Gate](https://www.researchgate.net/)

### :orange_book: Blogs

* [Medium](https://medium.com/machine-learning-world/tagged/computer-vision)
* [Towards Datascience](https://towardsdatascience.com/tagged/computer-vision)
* [Analytics Vidhya](https://www.analyticsvidhya.com/blog/tag/computer-vision/)
