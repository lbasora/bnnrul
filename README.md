# bnnrul (Bayesian Neural Networks (BNN) for Remaining Useful Life estimation)

This repository offers some initial thoughts, references and tools in preparation for an internship at ONERA to test BNN inference algorithms and techniques for RUL prediction on aeronautical systems.

The original code in this repository is now being further developed by the student at: https://github.com/arthurviens/bayesrul

### Ideas for work during the internship
To test the different BNN inference algorithms we can use the NASA CMAPSS and/or N-CMAPSS datasets.

For N-CMAPSS, [see 2021 PHM Conference Data Challenge](https://data.phmsociety.org/2021-phm-conference-data-challenge/). Winners: [paper1](https://papers.phmsociety.org/index.php/phmconf/article/view/3108), [paper2](https://papers.phmsociety.org/index.php/phmconf/article/view/3109), [paper3](https://papers.phmsociety.org/index.php/phmconf/article/view/3110) 

As there are several BNN frameworks available (TyXe, bayesian-torch...), it will be necessary to assess them to make a choice (see Tools subsection in the end). It would be nice to make some contribution as most of them are in an early stage of development.

### Datasets

NASA DASHlink:

1) CMAPSS dataset for turbofan engines

Saxena, Abhinav, Kai Goebel, Don Simon, et Neil Eklund. « Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation ». In 2008 International Conference on Prognostics and Health Management, 1‑9. Denver, CO, USA: IEEE, 2008. https://doi.org/10.1109/PHM.2008.4711414.
Classical benchmark dataset. Lots of published articles for RUL benchmarking including a few with BNN.

2) N-CMAPSS dataset for turbojet engines

Arias Chao, Manuel, Chetan Kulkarni, Kai Goebel, et Olga Fink. « Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics ». Data 6, nᵒ 1 (13 janvier 2021): 5. https://doi.org/10.3390/data6010005.
Very recent, bigger and more realistic dataset than CMAPSS.

### Code Examples
- For the CMAPSS dataset, some notebooks and python code are provided to generate the LMDB dataset as well as some examples on how to train/test models with pytorch-lightning and TyXe tools.  
- For N-CMAPSS only a notebook (ncmapss_example_data_loading_and_exploration.ipynb) is provided based on the original one by the N-CMAPSS authors. So, if N-CMAPSS is to be used, it will be necessary to start by adapting the CMAPSS code to also generate a LMDB dataset.

1) Create a conda environment with Pytorch and CUDA toolkit 11.3.
```sh
conda create --name bnnrul
conda activate bnnrul
mamba install pytorch cudatoolkit=11.3 -c pytorch
```

2) Install Jupyter Lab and related tools to create a jupyter kernel for the conda env
```sh
mamba install -c conda-forge jupyterlab_widgets
mamba install -c conda-forge ipywidgets
mamba install -c anaconda ipykernel
python -m ipykernel install --user --name=bnnrul
```

3) Install BNN frameworks: TyXe, bayesian-torch ...

4) Install bnnrul:
```sh
git clone git@github.com:lbasora/bnnrul.git
cd bnnrul
python setup.py install
```

### References BNN

BNN general reference for people familiar with deterministic deep learning:

3) Jospin, Laurent Valentin, Wray Buntine, Farid Boussaid, Hamid Laga, et Mohammed Bennamoun. « Hands-on Bayesian Neural Networks -- a Tutorial for Deep Learning Users ». ArXiv:2007.06823 [Cs, Stat], 14 juillet 2020. http://arxiv.org/abs/2007.06823.

Google DeepMind paper introducing Bayes by Backprop technique:

4) Blundell, Charles, Julien Cornebise, Koray Kavukcuoglu, et Daan Wierstra. « Weight Uncertainty in Neural Networks ». ArXiv:1505.05424 [Cs, Stat], 21 mai 2015. http://arxiv.org/abs/1505.05424.

Paper introducing Local Reparameterization Trick (LRT):

5) Kingma, Diederik P., Tim Salimans, et Max Welling. « Variational Dropout and the Local Reparameterization Trick ». ArXiv:1506.02557 [Cs, Stat], 20 décembre 2015. http://arxiv.org/abs/1506.02557.

Paper introducing Flipout technique:

6) Wen, Yeming, Paul Vicol, Jimmy Ba, Dustin Tran, et Roger Grosse. « Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches ». ArXiv:1803.04386 [Cs, Stat], 2 avril 2018. http://arxiv.org/abs/1803.04386.

Recent technique with few published material, so may be worth including it in a benchmark with Bayes by Backprop, LRT, flipout:

7) Pearce, T., Zaki, M., Brintrup, A., Anastassacos, N., and Neely, A. Uncertainty in neural networks: Bayesian
ensembling. International Conference on Artificial Intelligence and Statistics (AISTATS), 2020

Dropout based technique:

8) Gal, Yarin, et Zoubin Ghahramani. « Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning ». ArXiv:1506.02142 [Cs, Stat], 4 octobre 2016. http://arxiv.org/abs/1506.02142.


### References BNN based RUL estimation

C-MAPSS:

9) Caceres, J., Gonzalez, D., Zhou, T., & Droguett, E. L. A probabilistic Bayesian recurrent neural network for remaining useful life prognostics considering epistemic and aleatory uncertainties. Structural Control and Health Monitoring, 2021, vol. 28, no 10

10) Benker, Maximilian, Lukas Furtner, Thomas Semm, et Michael F. Zaeh. « Utilizing Uncertainty Information in Remaining Useful Life Estimation via Bayesian Neural Networks and Hamiltonian Monte Carlo ». Journal of Manufacturing Systems, decembre 2020, https://doi.org/10.1016/j.jmsy.2020.11.005

11) Huang, Dengshan, Rui Bai, Shuai Zhao, Pengfei Wen, Shengyue Wang, et Shaowei Chen. « Bayesian Neural Network Based Method of Remaining Useful Life Prediction and Uncertainty Quantification for Aircraft Engine ». In 2020 IEEE International Conference on Prognostics and Health Management (ICPHM), https://doi.org/10.1109/ICPHM49022.2020.9187044

12) Peng, Weiwen, Zhi-Sheng Ye, et Nan Chen. « Bayesian Deep-Learning-Based Health Prognostics Toward Prognostics Uncertainty ». IEEE Transactions on Industrial Electronics 67, nᵒ 3 (march 2020): 2283‑93. https://doi.org/10.1109/TIE.2019.2907440

Other systems:

13) Li, Gaoyang, Li Yang, Chi-Guhn Lee, Xiaohua Wang, et Mingzhe Rong. « A Bayesian Deep Learning RUL Framework Integrating Epistemic and Aleatoric Uncertainties ». IEEE Transactions on Industrial Electronics, 2020, 1‑1. https://doi.org/10.1109/TIE.2020.3009593


### Tools

These are some of the frameworks compatible with pytorch which can be used for BNN training/testing. **TyXE** or **bayesian-torch** seem a priori good options. 

**TyXE** based on Pyro is powerful but the learning curbe is more important than for **bayesian-torch**. The integration with **pytorch-lighning** is probably easier with **bayesian-torch** and the contributors seem more active.


1) [TyXE](https://github.com/TyXe-BDL/TyXe)

- BNNs for [Pyro](https://pyro.ai/) users. Pyro Tutorial: http://pyro.ai/examples/intro_long.html
- Linear and CNN based BNN implemented (For RNN based BNN with flipout see issue #6).
- Preliminary tests in cmapss_rul_linear_tyxe.ipynb. Issues with checkpointing, compatibility with latest pyro version, integration with pytorch-lightning not evident.
- It seems very powerful with pyro as backend, but requires time to learn pyro mechanisms (poutines).

2) [bayesian-torch](https://github.com/IntelLabs/bayesian-torch#installing-bayesian-torch)

- A library for BNN layers and uncertainty estimation in Deep Learning extending the core of PyTorch (developed by IntelLabs).
- Linear, CNN and RNN implemented.
- Preliminary tests in cmapss_rul_linear_bt.ipynb.
- It seems to integrate with pytorch-lightning easily.
- Lacks likelihood functions for data noise modelling and perhaps other aspects present already in TyXe.

3) [blitz](https://github.com/piEsposito/blitz-bayesian-deep-learning)

- A simple and extensible library to create BNN layers on PyTorch. 
- Its author is also contributing to bayesian-torch.
- Bayesian layers need to be manually defined unlike in bayesian-torch/TyXe which can turn deterministic NN into BNN automatically.
- Do not know what advantages it has compared to bayesian-torch (haven't investigated it).
- Used in Benker et al. [10]

4) [hamiltorch](https://github.com/AdamCobb/hamiltorch)

- PyTorch-based library for Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) and inference in Bayesian neural networks.
- Used in Benker et al. [10]




