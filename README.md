# Revisit the Stability of Vanilla Federated Learning Under Diverse Conditions

This is an official implementation of the following paper:
> Youngjoon Lee, Jinu Gong, Sun Choi,and Joonhyuk Kang.
**[Revisit the Stability of Vanilla Federated Learning Under Diverse Conditions](https://arxiv.org/abs/2502.19849)**  
_MICCAI 2025 (Accepted)_.

## Federated Learning Methods
This paper considers the following federated learning techniques:
- **FedAvg**: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)
- **FedProx**: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf)
- **FedDyn**: [Federated Learning Based on Dynamic Regularization](https://openreview.net/pdf?id=B7v4QMR6Z9w)
- **FedCM**: [FedCM: Federated Learning with Client-level Momentum](https://openreview.net/pdf?id=B7v4QMR6Z9w)
- **FedSAM**: [Generalized Federated Learning via Sharpness Aware Minimization](https://proceedings.mlr.press/v162/qu22a/qu22a.pdf)
- **FedGamma**: [Fedgamma: Federated learning with global sharpness-aware minimization](https://ieeexplore.ieee.org/abstract/document/10269141)
- **FedSpeed**: [FedSpeed: Larger Local Interval, Less Communication Round, and Higher Generalization Accuracy](https://openreview.net/pdf?id=bZjxxYURKT)
- **FedSMOO**: [Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape](https://proceedings.mlr.press/v202/sun23h.html)

## Docker Image
`docker pull gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest`

## Dataset
- Blood cell classification dataset ([A dataset of microscopic peripheral blood cell images for development of automatic recognition systems](https://www.sciencedirect.com/science/article/pii/S2352340920303681))
- Skin lesion classification dataset ([The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://www.nature.com/articles/sdata2018161))

## Experiments

To run the 'Impact of Stability' experiment:
Run `bash shell/exp1.sh`

To run the 'Impact of Model Diversity' experiment:
Run `bash shell/exp2.sh`

To run the 'Impact of Hyperparameter Selection' experiment:
Run `bash shell/exp3.sh`

## Citation
If this codebase can help you, please cite our paper: 
```bibtex
@article{lee2025revisit,
  title={Revisit the Stability of Vanilla Federated Learning Under Diverse Conditions},
  author={Lee, Youngjoon and Gong, Jinu and Choi, Sun and Kang, Joonhyuk},
  journal={arXiv preprint arXiv:2502.19849},
  year={2025}
}
```

## References
This repository draws inspiration from:
- https://github.com/woodenchild95/FL-Simulator
- https://github.com/ZiyaoLi/fast-kan
