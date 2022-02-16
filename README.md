# bnnrul (Bayesian Neural Networks for Remaining Useful Life estimation)

Tools to test BNN inference algorithms and techniques to predict RUL on aeronautical systems.

### Ideas for research work
- Benchmark with NASA CMAPSS [1] dataset complementary to the existing ones (e.g. Caceres et al. [9]) by using other inference algorithms.
-  Benchmark with NASA N-CMAPSS [2] dataset on the recent big NASA dataset Arias et al.
Few publications yet with N-CMAPSS, so it might be an opportunity to publish.
    For N-CMAPSS, [see 2021 PHM Conference Data Challenge](https://data.phmsociety.org/2021-phm-conference-data-challenge/). Winners: [paper1](https://papers.phmsociety.org/index.php/phmconf/article/view/3108), [paper2](https://papers.phmsociety.org/index.php/phmconf/article/view/3109), [paper3](https://papers.phmsociety.org/index.php/phmconf/article/view/3110)
- Contribution to [TyXE](https://github.com/TyXe-BDL/TyXe) project (e.g. Flipout RNN - see issue #6)

### Datasets

NASA DASHlink:

1) CMAPSS dataset for turbofan engines (/data/data/nasa/CMAPSSData.zip)
Saxena, Abhinav, Kai Goebel, Don Simon, et Neil Eklund. « Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation ». In 2008 International Conference on Prognostics and Health Management, 1‑9. Denver, CO, USA: IEEE, 2008. https://doi.org/10.1109/PHM.2008.4711414.
Classical benchmark dataset. Lots of published articles for RUL benchmarking including a few with BNN.

2) N-CMAPSS dataset for turbojet engines (/data/data/nasa/CMAPSSData2.zip)
Arias Chao, Manuel, Chetan Kulkarni, Kai Goebel, et Olga Fink. « Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics ». Data 6, nᵒ 1 (13 janvier 2021): 5. https://doi.org/10.3390/data6010005.
Very recent, bigger and more realistic dataset than CMAPSS.


### References BNN

3) Jospin, Laurent Valentin, Wray Buntine, Farid Boussaid, Hamid Laga, et Mohammed Bennamoun. « Hands-on Bayesian Neural Networks -- a Tutorial for Deep Learning Users ». ArXiv:2007.06823 [Cs, Stat], 14 juillet 2020. http://arxiv.org/abs/2007.06823.

BNN general reference for people familiar with deterministic deep learning.

4) Blundell, Charles, Julien Cornebise, Koray Kavukcuoglu, et Daan Wierstra. « Weight Uncertainty in Neural Networks ». ArXiv:1505.05424 [Cs, Stat], 21 mai 2015. http://arxiv.org/abs/1505.05424.

Google DeepMind paper introducing Bayes by Backprop technique.

5) Kingma, Diederik P., Tim Salimans, et Max Welling. « Variational Dropout and the Local Reparameterization Trick ». ArXiv:1506.02557 [Cs, Stat], 20 décembre 2015. http://arxiv.org/abs/1506.02557.

Paper introducing Local Reparameterization (LRT) Trick.

6) Wen, Yeming, Paul Vicol, Jimmy Ba, Dustin Tran, et Roger Grosse. « Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches ». ArXiv:1803.04386 [Cs, Stat], 2 avril 2018. http://arxiv.org/abs/1803.04386.

Paper introducing Flipout technique.

7) Pearce, T., Zaki, M., Brintrup, A., Anastassacos, N., and Neely, A. Uncertainty in neural networks: Bayesian
ensembling. International Conference on Artificial Intelligence and Statistics (AISTATS), 2020

Recent technique with few published material, so may be worth including it in a benchmark with Bayes by Backprop, LRT, flipout. 

8) Gal, Yarin, et Zoubin Ghahramani. « Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning ». ArXiv:1506.02142 [Cs, Stat], 4 octobre 2016. http://arxiv.org/abs/1506.02142.


### References BNN based RUL estimation
9) Caceres, Jose, Danilo Gonzalez, Taotao Zhou, et Enrique Lopez Droguett. « A Probabilistic Bayesian Recurrent Neural Network for Remaining Useful Life Prognostics Considering Epistemic and Aleatory Uncertainties », s. d., 21.

10) Huang, Dengshan, Rui Bai, Shuai Zhao, Pengfei Wen, Shengyue Wang, et Shaowei Chen. « Bayesian Neural Network Based Method of Remaining Useful Life Prediction and Uncertainty Quantification for Aircraft Engine ». In 2020 IEEE International Conference on Prognostics and Health Management (ICPHM), 1‑8. Detroit, MI, USA: IEEE, 2020. https://doi.org/10.1109/ICPHM49022.2020.9187044

11) Li, Gaoyang, Li Yang, Chi-Guhn Lee, Xiaohua Wang, et Mingzhe Rong. « A Bayesian Deep Learning RUL Framework Integrating Epistemic and Aleatoric Uncertainties ». IEEE Transactions on Industrial Electronics, 2020, 1‑1. https://doi.org/10.1109/TIE.2020.3009593.

12) Peng, Weiwen, Zhi-Sheng Ye, et Nan Chen. « Bayesian Deep-Learning-Based Health Prognostics Toward Prognostics Uncertainty ». IEEE Transactions on Industrial Electronics 67, nᵒ 3 (mars 2020): 2283‑93. https://doi.org/10.1109/TIE.2019.2907440.


### Tools
1) [pyro](https://pyro.ai/): Deep Universal Probabilistic Programming
Tutorial: http://pyro.ai/examples/intro_long.html

2) [TyXE](https://github.com/TyXe-BDL/TyXe): Pyro-based BNNs for Pytorch users
Linear and CNN based BNN implemented.
For RNN based BNN with flipout see issue #6

3) [Edward2](https://github.com/google/edward2):
A simple probabilistic programming language. Same purpose than pyro+tyxe but with tensorflow, numpy or jax as backends.

3) https://github.com/kkangshen/bayesian-deep-rul





