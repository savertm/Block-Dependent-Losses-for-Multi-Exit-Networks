# Improving Low-Latency Predictions in Multi-Exit Neural Networks via Block-Dependent Losses published in IEEE Transactions on Neural Networks and Learning Systems (TNNLS)


This repo contains the PyTorch implementation of our TNNLS paper, [Improving Low-Latency Predictions in Multi-Exit Neural Networks via Block-Dependent Losses](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10155768)

**Abstract:** As the size of a model increases, making predictions using deep neural networks (DNNs) is becoming more computationally expensive. Multi-exit neural network is one promising solution that can flexibly make anytime predictions via early exits, depending on the current test-time budget which may vary over time in practice (e.g., selfdriving cars with dynamically changing speeds). However, the prediction performance at the earlier exits is generally much lower than the final exit, which becomes a critical issue in low-latency applications having a tight test-time budget. Compared to the previous works where each block is optimized to minimize the losses of all exits simultaneously, in this work, we propose a new method for training multi-exit neural networks by strategically imposing different objectives on individual blocks. The proposed idea based on grouping and overlapping strategies improves the prediction performance at the earlier exits while not degrading the performance of later ones, making our scheme to be more suitable for low-latency applications. Extensive experimental results on both image classification and semantic segmentation confirm the advantage of our approach. The proposed idea does not require any modifications in the model architecture and can be easily combined with existing strategies aiming to improve the performance of multi-exit neural networks.


## Requirements

This code was tested on the following environments:

* Ubuntu 18.04
* Python 3.7.13
* PyTorch 1.12.0
* CUDA 11.6

You can install all necessary packages from requirements.txt

```
pip install -r requirements.txt
```

## Experiments

* Experiments can be conducted on two image classification datasets: CIFAR-100, ImageNet. 

### How to Run

* All parameters required for the experiment are described in ```args.py```. Please see the python file for a detailed description of the parameters.

```bash

# Cifar-100 dataset

bash train.sh

# ImageNet dataset

bash train_imagenet.sh

```


## Citation

To cite our paper in your papers, please use the following bibtex entry.

```
@article{han2023improving,
  title={Improving Low-Latency Predictions in Multi-Exit Neural Networks via Block-Dependent Losses},
  author={Han, Dong-Jun and Park, Jungwuk and Ham, Seokil and Lee, Namjin and Moon, Jaekyun},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement

Our code is built upon the implementations at https://github.com/kalviny/MSDNet-PyTorch
