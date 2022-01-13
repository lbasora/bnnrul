# bnnrul (Bayesian Neural Networks for Remaining Useful Life estimation)

Tools to test BNN inference algorithms and techniques to predict RUL on aeronautical systems.

Ideas to explore/publish:
- Benchmark complementary to Caceres et al. [6] with other inference algorithms, etc.
- Test on the recent big NASA dataset Arias et al. [9]

### References BNN
General

1) Jospin, Laurent Valentin, Wray Buntine, Farid Boussaid, Hamid Laga, et Mohammed Bennamoun. « Hands-on Bayesian Neural Networks -- a Tutorial for Deep Learning Users ». ArXiv:2007.06823 [Cs, Stat], 14 juillet 2020. http://arxiv.org/abs/2007.06823.

Main advances

2) Blundell, Charles, Julien Cornebise, Koray Kavukcuoglu, et Daan Wierstra. « Weight Uncertainty in Neural Networks ». ArXiv:1505.05424 [Cs, Stat], 21 mai 2015. http://arxiv.org/abs/1505.05424.

3) Kingma, Diederik P., Tim Salimans, et Max Welling. « Variational Dropout and the Local Reparameterization Trick ». ArXiv:1506.02557 [Cs, Stat], 20 décembre 2015. http://arxiv.org/abs/1506.02557.

4) Wen, Yeming, Paul Vicol, Jimmy Ba, Dustin Tran, et Roger Grosse. « Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches ». ArXiv:1803.04386 [Cs, Stat], 2 avril 2018. http://arxiv.org/abs/1803.04386.

Optional

4) Gal, Yarin, et Zoubin Ghahramani. « Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning ». ArXiv:1506.02142 [Cs, Stat], 4 octobre 2016. http://arxiv.org/abs/1506.02142.

5) Pearce, Tim, Felix Leibfried, Alexandra Brintrup, Mohamed Zaki, et Andy Neely. « Uncertainty in Neural Networks: Approximately Bayesian Ensembling », s. d., 10.

### References BNN based RUL estimation
6) Caceres, Jose, Danilo Gonzalez, Taotao Zhou, et Enrique Lopez Droguett. « A Probabilistic Bayesian Recurrent Neural Network for Remaining Useful Life Prognostics Considering Epistemic and Aleatory Uncertainties », s. d., 21.

7) Huang, Dengshan, Rui Bai, Shuai Zhao, Pengfei Wen, Shengyue Wang, et Shaowei Chen. « Bayesian Neural Network Based Method of Remaining Useful Life Prediction and Uncertainty Quantification for Aircraft Engine ». In 2020 IEEE International Conference on Prognostics and Health Management (ICPHM), 1‑8. Detroit, MI, USA: IEEE, 2020. https://doi.org/10.1109/ICPHM49022.2020.9187044

8) Li, Gaoyang, Li Yang, Chi-Guhn Lee, Xiaohua Wang, et Mingzhe Rong. « A Bayesian Deep Learning RUL Framework Integrating Epistemic and Aleatoric Uncertainties ». IEEE Transactions on Industrial Electronics, 2020, 1‑1. https://doi.org/10.1109/TIE.2020.3009593.

### Datasets

NASA DASHlink:

9) Classical smaller CMAPSS dataset for turbofan engines (/data/data/nasa/CMAPSSData.zip)
Saxena, Abhinav, Kai Goebel, Don Simon, et Neil Eklund. « Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation ». In 2008 International Conference on Prognostics and Health Management, 1‑9. Denver, CO, USA: IEEE, 2008. https://doi.org/10.1109/PHM.2008.4711414.

10) Recent big CMAPSS dataset for turbojet engines (/data/data/nasa/CMAPSSData2.zip)
Arias Chao, Manuel, Chetan Kulkarni, Kai Goebel, et Olga Fink. « Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics ». Data 6, nᵒ 1 (13 janvier 2021): 5. https://doi.org/10.3390/data6010005.


### Tools
1) pyro: Deep Universal Probabilistic Programming
Tutorial: http://pyro.ai/examples/intro_long.html

2) TyXE: Pyro-based BNNs for Pytorch users
Linear and CNN based BNN implemented.
For RNN based BNN with flipout see issue #6

3) https://github.com/kkangshen/bayesian-deep-rul





