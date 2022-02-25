<h1><center> Deep Learning Reference </center></h1>
<hr/>

## Table of Contents

- [Frameworks](#Frameworks)
    * [Low-level Deep Learning Frameworks](#Low-level-Deep-Learning-Frameworks)
        + [<img src="https://pytorch.org/favicon.ico" width="32" />PyTorch](#-Pytorch)
        + [<img src="https://www.tensorflow.org/favicon.ico" width="32" />TensorFlow](#-TensorFlow)
        + [<img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/>Caffe](#-Caffe)
        + [<img src="https://pjreddie.com/static/icon.png" width="32"/>Darknet](#-Darknet)
        + [<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png" width="32"/>MXNet](#-MXNet)
     * [High-level Deep Learning Frameworks](#High-level-Deep-Learning-Frameworks)
        + [<img src="https://keras.io/favicon.ico" width="32">Keras](#-Keras)
        + [<img src="https://assets.website-files.com/5f76c986da6f6011315a6c45/5f7b485594e5335f56984a3c_Lightning_FavIcon.png" width="32">PyTorch Lightning](#-PyTorch-Lightning)
        + [<img src="https://pytorch.org/ignite/_static/ignite_logomark.svg" width=32 />PyTorch Ignite](#-PyTorch-Ignite)
    * [Hyper-Parameter Optimization Frameworks](#Hyper-Parameter-Optimization-Frameworks)
        + [<img src="https://optuna.org/assets/img/favicon.ico" width=32 />Optuna](#-Optuna)
        + [<img src="https://docs.ray.io/en/latest/_static/favicon.ico" width=32 /> Ray](#-Ray)
        + [Hyperopt](#Hyperopt)
    * [Experiment Tracking](#Experiment-Tracking)
    	+ [<img src="https://www.tensorflow.org/favicon.ico" width="32" />TensorBoard](#-TensorBoard)
    	+ [<img src="https://mlflow.org/favicon.ico" width=32 />MLFlow](#-MLFlow)
    	+ [<img src="https://i0.wp.com/neptune.ai/wp-content/uploads/2019/03/cropped-Artboard-12.png?fit=32%2C32&#038;ssl=1" width=32 />Neptune](#Neptune)
    	+ [<img src="https://wandb.ai/favicon.ico" width=32 />Weights & Biases](#Weight-&-Biases)
+ [Optimizers](#Optimizers)
+ [Loss functions](#Loss-functions)
  + [Classification](#Classification)
  + [Regression](#Regression)
  + [Distribution Regression](#Distribution-Regression)
  + [Embedding](#Embedding)
- [Material](#Material)
  * [Books](#Books)
  * [Papers](#Papers)
  * [Blogs](#Blogs)

## Frameworks

### Low-level Deep Learning Frameworks

Frameworks that provide features like

* **Tensor computation**
* **Automatic Differentiation**
* **Computational Graphs**
* **GPU Support**

| Framework                                                    | Creator       | Python             | C/C++              | R                  | Java               | Javascript         |
| ------------------------------------------------------------ | ------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| [<img src="https://pytorch.org/favicon.ico" width="32" />PyTorch](#-Pytorch) | Facebook      | :heavy_check_mark: | :heavy_check_mark: |                    |                    | :heavy_check_mark: |
| [<img src="https://www.tensorflow.org/favicon.ico" width="32" />Tensorflow](#-TensorFlow) | Google        | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |                   ||||
| [<img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/>Caffe](#-Caffe) | UC Berkeley   | :heavy_check_mark: | :heavy_check_mark: |                    |                    |                    |
| [<img src="https://pjreddie.com/static/icon.png" width="32"/>Darknet](#-Darknet) | Joseph Redmon |                    | :heavy_check_mark: |                    |                    |                    |
| [<img src="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mxnet-icon.png" width="32"/>MXNet](#-MXNet) | Apache        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |


#### <img src="https://pytorch.org/favicon.ico" width="32" /> Pytorch

> A Python package that provides
>   * Tensor computation (like NumPy) with strong GPU acceleration
>   * Deep neural networks built on a tape-based autograd system

* :house: **Main page**: https://pytorch.org/
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/pytorch/pytorch ![GitHub stars](https://img.shields.io/github/stars/pytorch/pytorch?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/torch ![PyPi version](https://badgen.net/pypi/v/torch/)]
* :blue_book: **Docs**: https://pytorch.org/docs/stable/index.html
*  **Docker**:
    * [DockerHub](https://hub.docker.com/r/pytorch/pytorch) ![docker pulls](https://img.shields.io/docker/pulls/pytorch/pytorch.svg?style=social)
    * [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

#### <img src="https://www.tensorflow.org/favicon.ico" width="32" /> TensorFlow 

> An end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

* :house: **Main page**: https://www.tensorflow.org/?hl=es-419
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/tensorflow/tensorflow  ![GitHub stars](https://img.shields.io/github/stars/tensorflow/tensorflow?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/tensorflow [![PyPi version](https://badgen.net/pypi/v/tensorflow/)](https://pypi.org/project/tensorflow)
* :blue_book: **Docs**: https://www.tensorflow.org/api_docs
* **Docker**:
    * [DockerHub](https://hub.docker.com/r/tensorflow/tensorflow) ![docker pulls](https://img.shields.io/docker/pulls/tensorflow/tensorflow.svg?style=social)
    * [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)

#### <img src="https://caffe.berkeleyvision.org/images/caffeine-icon.png" width="32"/> Caffe

> A deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (BAIR) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license.

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

### High-level Deep Learning Frameworks

Frameworks that is built on top of a low-level Framework that provide high-level features to facilitate deployment, distributed/non-distributed training, model design, etc.

| Framework | Creator | Base Framework |
| --------- | ------- | -------------- |
| [<img src="https://keras.io/favicon.ico" width="32">Keras](#-Keras) | Google     | [<img src="https://www.tensorflow.org/favicon.ico" width="32" />TensorFlow](#-TensorFlow) |
| [<img src="https://assets.website-files.com/5f76c986da6f6011315a6c45/5f7b485594e5335f56984a3c_Lightning_FavIcon.png" width="32">PyTorch Lightning](#-PyTorch-Lightning) | CILVR     | [<img src="https://pytorch.org/favicon.ico" width="32" />PyTorch](#-Pytorch) |
| [<img src="https://pytorch.org/ignite/_static/ignite_logomark.svg" width=32 />PyTorch Ignite](#-PyTorch-Ignite) | Facebook  | [<img src="https://pytorch.org/favicon.ico" width="32" />PyTorch](#-Pytorch)  |

#### <img src="https://keras.io/favicon.ico" width="32"> Keras

> A deep learning API written in Python, running on top of the machine learning platform [TensorFlow](https://github.com/tensorflow/tensorflow). It was developed with a focus on enabling fast experimentation. *Being able to go from idea to result as fast as possible is key to doing good research*          

* :house: **Main page**: https://keras.io/
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/keras-team/keras ![GitHub stars](https://img.shields.io/github/stars/keras-team/keras?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 /> **PyPI**:  https://pypi.org/project/keras [![PyPi version](https://badgen.net/pypi/v/keras/)](https://pypi.org/project/keras)
* :blue_book: **Docs**: https://keras.io/api/

#### <img src="https://assets.website-files.com/5f76c986da6f6011315a6c45/5f7b485594e5335f56984a3c_Lightning_FavIcon.png" width="32"> PyTorch Lightning

> A PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.

- :house: **Main page**: https://www.pytorchlightning.ai/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/PyTorchLightning/pytorch-lightning ![GitHub stars](https://img.shields.io/github/stars/PyTorchLightning/pytorch-lightning?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/pytorch-lightning [![PyPi version](https://badgen.net/pypi/v/pytorch-lightning/)](https://pypi.org/project/pytorch-lightning)
- :blue_book: **Docs**: https://pytorch-lightning.readthedocs.io/en/stable/

#### <img src="https://pytorch.org/ignite/_static/ignite_logomark.svg" width=32 /> PyTorch Ignite

> A high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.

* :house: **Main page**: https://pytorch.org/ignite
* ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/pytorch/ignite  ![GitHub stars](https://img.shields.io/github/stars/pytorch/ignite?style=social)
* <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**: https://pypi.org/project/pytorch-ignite/ [![PyPi version](https://badgen.net/pypi/v/pytorch-ignite/)](https://pypi.org/project/pytorch-ignite/)

### Hyper-Parameter Optimization Frameworks

Frameworks to fine-tune models hyper-parameters during model design

#### <img src="https://optuna.org/assets/img/favicon.ico" width=32 /> Optuna

> An automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters.

- :house: **Main page**: https://optuna.org/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/optuna/optuna ![GitHub stars](https://img.shields.io/github/stars/optuna/optuna?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/optuna [![PyPi version](https://badgen.net/pypi/v/optuna/)](https://pypi.org/project/optuna)
- :blue_book: **Docs**: https://optuna.readthedocs.io/en/stable/tutorial/index.html

#### <img src="https://docs.ray.io/en/latest/_static/favicon.ico" width=32 /> Ray

> An open source framework that provides a simple, universal API for building distributed applications. Ray is packaged with RLlib, a scalable reinforcement learning library, and Tune, a scalable hyperparameter tuning library.

- :house: **Main page**: https://www.ray.io/ray-tune
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/ray-project/ray ![GitHub stars](https://img.shields.io/github/stars/ray-project/ray?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/ray [![PyPi version](https://badgen.net/pypi/v/ray/)](https://pypi.org/project/ray)
- :blue_book: **Docs**: https://docs.ray.io/en/master/tune/index.html

#### Hyperopt

> A Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.

- :house: **Main page**: http://hyperopt.github.io/hyperopt/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/hyperopt/hyperopt ![GitHub stars](https://img.shields.io/github/stars/hyperopt/hyperopt?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/hyperopt [![PyPi version](https://badgen.net/pypi/v/hyperopt/)](https://pypi.org/project/hyperopt)
- :blue_book: **Docs**: http://hyperopt.github.io/hyperopt/#documentation

### Experiment tracking

Framework to track experiments during model development.

#### <img src="https://www.tensorflow.org/favicon.ico" width="32" /> TensorBoard

> A Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions

- :house: **Main page**: https://www.tensorflow.org/tensorboard
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/tensorflow/tensorboard ![GitHub stars](https://img.shields.io/github/stars/tensorflow/tensorboard?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/tensorboard [![PyPi version](https://badgen.net/pypi/v/tensorboard/)](https://pypi.org/project/tensorboard)
- :blue_book: **Tutorials**: https://www.tensorflow.org/tensorboard/get_started

#### <img src="https://mlflow.org/favicon.ico" width=32 /> MLFlow

> A platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models. MLflow offers a set of lightweight APIs that can be used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc), wherever you currently run ML code (e.g. in notebooks, standalone applications or the cloud)

- :house: **Main page**: https://www.mlflow.org
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/mlflow/mlflow ![GitHub stars](https://img.shields.io/github/stars/mlflow/mlflow?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/mlflow [![PyPi version](https://badgen.net/pypi/v/mlflow/)](https://pypi.org/project/mlflow)
- :blue_book: **Tutorials**: https://mlflow.org/docs/latest/tracking.html

#### <img src="https://i0.wp.com/neptune.ai/wp-content/uploads/2019/03/cropped-Artboard-12.png?fit=32%2C32&#038;ssl=1" width=32 /> Neptune

> Neptune is a lightweight solution designed for:
>
> - **Experiment tracking:** log, display, organize, and compare ML experiments in a single place.
> - **Model registry:** version, store, manage, and query trained models, and model-building metadata.
> - **Monitoring ML runs live:** record and monitor model training, evaluation, or production runs live.  

- :house: **Main page**: https://neptune.ai/
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/neptune-ai/neptune-client ![GitHub stars](https://img.shields.io/github/stars/neptune-ai/neptune-client?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/neptune-client [![PyPi version](https://badgen.net/pypi/v/neptune-client/)](https://pypi.org/project/neptune-client)
- :blue_book: **Tutorials**: https://docs.neptune.ai/#experiment-tracking

#### <img src="https://wandb.ai/favicon.ico" width=32 /> Weights & Biases

> A tool to build better models faster. Track and visualize all the pieces of your machine learning pipeline, from datasets to production models.
>
> - Quickly identify model regressions. Use W&B to visualize results in real time, all in a central dashboard.
> - Focus on the interesting ML. Spend less time manually tracking results in spreadsheets and text files.
> - Capture dataset versions with W&B Artifacts to identify how changing data affects your resulting models.
> - Reproduce any model, with saved code, hyperparameters, launch commands, input data, and resulting model weights.

- :house: **Main page**: https://wandb.ai/site
- ![GitHub icon](https://github.githubassets.com/images/modules/site/icons/footer/github-mark.svg) **GitHub**: https://github.com/wandb/client ![GitHub stars](https://img.shields.io/github/stars/wandb/client?style=social)
- <img src="https://pypi.org/static/images/logo-small.95de8436.svg" width=32 />**PyPI**:  https://pypi.org/project/wandb [![PyPi version](https://badgen.net/pypi/v/wandb/)](https://pypi.org/project/wandb)
- :blue_book: **Tutorials**: https://wandb.ai/site/tutorials
- **Docker**: [DockerHub](https://hub.docker.com/r/wandb/local) ![docker pulls](https://img.shields.io/docker/pulls/wandb/local.svg?style=social)

## Optimizers

Algorithms to minimize a loss function by updating model weights.

| Name | HyperParameters  | Name | Paper |
|----------|--------------|----------|------|
| **SGD** | Momentum, Dampering | Step Gradient Descent with Nesterov Momentum | |
| **ASGD** | lambda, alpha, t0 | Averaged Stochastic Gradient Descent | [Acceleration of Stochastic Approximation by Averaging](https://www.researchgate.net/publication/236736831_Acceleration_of_Stochastic_Approximation_by_Averaging) |
| **Adam** | betas | Adam | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) |
| **AdaGrad** | lr_decay | Adaptative Subgradient | [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.org/papers/v12/duchi11a.html) |
| **AdaDelta** | rho | AdaDelta | [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701) |
| **AdaMax** | betas | AdaMax | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) |
| **AdamW** | betas | AdamW | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) |
| **NAdam** | betas | AMSGrad Adam | [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237) |
| **RProp** | Etaminus, Etaplus, Step Size | Resilient Backpropagation |   |
| **RMSProp** | Alpha, Centered | RMSProp |[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) |
| **L-BFGS** | max_iter, min_iter, tolerance_change, history_size, | Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm | |

## Loss functions

Cost function maps how well a model is behaving on train & test data.

In all the following Loss functions "x" is the predicted values whereas "y" is the target

### Classification

| Classification Type   | Name                                    | Target constraint           | Formula                                                      |
| --------------------- | --------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| **MultiLabel**        | Binary Cross Entropy Loss (with logits) | y in [0..1]                 | <img src="https://latex.codecogs.com/svg.latex?y%20%5Ccdot%20%5Clog%20%5Csigma%28x%29%20+%20%281-y%29%20%5Ccdot%20%5Clog%20%281-%5Csigma%28x%29%29" /> |
| **Multiclass**        | Cross Entropy Loss (with logits)        | y_i is a prob. distribution | <img  src="https://latex.codecogs.com/svg.image?%5Csum%5EC_%7Bi=1%7D%20y_i%20%5Ccdot%20%5Clog%20%5Cfrac%7Be%5E%7Bx_i%7D%7D%7B%5Csum%5EC_%7Bj=1%7D%20e%5E%7Bx_j%7D%7D" /> |


### Regression ###

| Name        | Formula                                         |
| ----------- | ----------------------------------------------- |
| **L1 Loss** (**MAE**) | <img src="https://latex.codecogs.com/svg.image?%7Cx-y%7C"/> |
| **L2 Loss** (**MSE**) | <img src="https://latex.codecogs.com/svg.image?(x-y)^2" title="(x-y)^2"/> |
| **Smooth L1 Loss** | <img src="https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5Cfrac%7B1%7D%7B2%7D%5Cfrac%7B(x%20-%20y)%5E2%7D%7B%5Cbeta%7D,%20&%20%5Ctext%7Bif%20%7D%20%7Cx%20-%20y%7C%20%3C%20%5Cbeta%20%5C%5C%7Cx%20-%20y%7C%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Cbeta,%20&%20%5Ctext%7Botherwise%20%7D%5Cend%7Bmatrix%7D%5Cright." /> |
| **Uber Loss** | <img src="https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%5Cfrac%7B1%7D%7B2%7D%20(x%20-%20y)%5E2,%20&%20%5Ctext%7Bif%20%7D%20%7Cx%20-%20y%7C%20%3C%20%5Cdelta%5C%5C%5Cdelta%20%5Ccdot%20(%7Cx%20-%20y%7C%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Cdelta),%20&%20%5Ctext%7Botherwise%20%7D%5Cend%7Bmatrix%7D%5Cright." /> |

### Distribution regression  ###

| Name | Target distribution | Formula |
|----------|---------------------------|--------------|
| **Binary Cross Entropy** | Bernoulli(x) | <img src="https://latex.codecogs.com/svg.image?y%20%5Clog%20x%20&plus;%20(1-y)%20%5Clog%20(1-x)" /> |
| **Cross Entropy Loss** | Dirchlet((x_1,...,x_C)) | <img src="https://latex.codecogs.com/svg.image?%5Csum%5EC_%7Bi=1%7D%20y_i%20%5Ccdot%20%5Clog%20x_i" /> |
| **Poisson Negative Log Likelihood** | Poisson(x) | <img src="https://latex.codecogs.com/svg.image?x-y%5Ccdot%20%5Clog%20x" /> |
| **Gaussian Negative Log Likelihood distribution** | Gaussian(x, v) | <img src="https://latex.codecogs.com/svg.image?%5Cfrac%7B1%7D%7B2%7D%20%5CBig%5B%20%5Clog%20%5Cmax%5Cleft%5C%7Bv,%20%5Cepsilon%5Cright%5C%7D%20&plus;%20%5Cfrac%7B(x-y)%5E2%7D%7B%5Cmax%5Cleft%5C%7Bv,%20%5Cepsilon%5Cright%5C%7D%7D%20%5CBig%5D" /> |
### Embedding

| Name                      | Formula                                                      |
| ------------------------- | ------------------------------------------------------------ |
| **Margin Ranking Loss**   | <img src="https://latex.codecogs.com/svg.image?%5Cmax%20%5Cleft%5C%7B%200,%20-%20y%20%5Ccdot%20(x_1-x_2)%20&plus;%20%5CDelta%20%5Cright%5C%7D" /> |
| **Cosine Embedding Loss** | <img src="https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D1-%5Ccos(x_1,%20x_2)%20&%20%5Cmbox%7Bif%20%7D%20y%20=%201%20%5C%5C%5Cmax%20%5Cleft%5C%7B%200,%20%20%5Ccos(x_1,%20x_2)%20-%20%5CDelta%20%5Cright%5C%7D%20&%20%5Cmbox%7Bif%20%7D%20y%20=%20-1%20%20%5C%5C%5Cend%7Bmatrix%7D%5Cright." /> |
| **Triplet Margin Loss**   | <img src="https://latex.codecogs.com/svg.image?%5Cmax%5Cleft%5C%7B0,%20d(x,y_1)-d(x,y_2)%20&plus;%20%5CDelta%20%5Cright%5C%7D" /> |
| **Hinge Embedding Loss**  | <img src="https://latex.codecogs.com/svg.image?%5Cleft%5C%7B%5Cbegin%7Bmatrix%7Dx%20&%20%5Cmbox%7Bif%20%7D%20y%20=%201%20%5C%5C%5Cmax%20%5Cleft%5C%7B%200,%20%5CDelta-x%20%5Cright%5C%7D%20&%20%5Cmbox%7Bif%20%7D%20y=-1%20%5C%5C%5Cend%7Bmatrix%7D%5Cright." /> |



## Material

### :books: Books

* [Deep Learning book](https://www.deeplearningbook.org/)

### :page_facing_up: Papers

* [Arxiv](https://arxiv.org/)
* [PapersWithCode](https://paperswithcode.com/)
* [Research Gate](https://www.researchgate.net/)

### :orange_book: Blogs

* [Medium](https://medium.com/machine-learning-world/tagged/deep-learning)
* [Towards Datascience](https://towardsdatascience.com/tagged/deep-learning)
* [Analytics Vidhya](https://www.analyticsvidhya.com/blog/category/deep-learning/)