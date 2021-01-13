# DeepSpectralRanking
The code in this repository implements the algorithms and experiments in the following paper:  
> I. Yildiz, J. Dy, D. Erdogmus, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Deep Spectral Ranking”, AISTATS, San Diego, 2021

To begin with, install all package dependencies via `keras_gpu_env_3_6.yml`. The following command executes this in an Anaconda environment:
```
conda env create -f keras_gpu_env_3_6.yml
```

We implement three ranking regression algorithms that regress Plackett-Luce scores via deep neural network (DNN) models from ranking observations via Maximum Likelihood Estimation. 
- Deep Spectral Ranking-KL (DSR-KL) is a fast spectral algorithm based on ADMM with Kullback-Leibler (KL) Divergence proximal penalty (implemented in `deep_kl_admm.py`). This is the main contribution of the paper and has the best performance in terms of both convergence time and predictions. 
- DSR-l2 is another fast spectral algorithm based on ADMM with standard l2-norm proximal penalty (implemented in `deep_l2_admm.py`)
- Siamese network that has the same number of base networks as the number of samples in each ranking observation (implemented in `siamese_network.py`)

We also implement the shallow counterparts of DSR-KL and DSR-l2 where scores are regressed by affine functions of features rather than DNN functions (see `shallow_model_competitors.py`).



# Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> I. Yildiz, J. Dy, D. Erdogmus, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Deep Spectral Ranking”, AISTATS, San Diego, 2021

# Acknowledgements
Our work is supported by NIH (R01EY019474), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).
