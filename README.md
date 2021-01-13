# DeepSpectralRanking
The code in this repository implements the algorithms and experiments in the following paper:  
> I. Yildiz, J. Dy, D. Erdogmus, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Deep Spectral Ranking”, AISTATS, San Diego, 2021

To begin with, install all package dependencies via `keras_gpu_env_3_6.yml`. The following command executes this in an Anaconda environment:
```
conda env create -f keras_gpu_env_3_6.yml
```

We implement three ranking regression algorithms that regress Plackett-Luce scores via deep neural network (DNN) models from ranking observations via Maximum Likelihood Estimation. 
- Deep Spectral Ranking-KL (DSR-KL) is a fast spectral algorithm based on ADMM with Kullback-Leibler (KL) Divergence proximal penalty (implemented in `deep_kl_admm.py`). This is the *main contribution of the paper* and has the best performance in terms of both convergence time and predictions. 
- DSR-l2 is another fast spectral algorithm based on ADMM with standard l2-norm proximal penalty (implemented in `deep_l2_admm.py`)
- Siamese network that has the same number of base networks as the number of samples in each ranking observation (implemented in `siamese_network.py`)

We also implement the shallow counterparts of DSR-KL and DSR-l2 where scores are regressed by affine functions of features rather than DNN functions (see `shallow_model_competitors.py`).

We evaluate these algorithms on real-life datasets, where preprocessing is handled in `preprocessing_image_datasets.py`. For datasets with a global total ranking rather than expert generated ranking labels (c.f. ICLR paper ratings, IMDB movie ratings, Movehub-Cost and Movehub-Quality city ratings), we generated all possible K-way rankings. Then, to simulate real-life labeling noise, we randomly select a fraction of the generated rankings and add noise by permuting each ranking. We perform 5-fold cross validation and partition each dataset into training, validation, and test sets in two ways. In rank partitioning, we partition the dataset w.r.t. ranking observations, using 80% of all observations for training, 10% for validation, and the remaining 10% for testing. In sample partitioning, we partition samples, using 80% of all samples for training, 10% for validation, and the remaining 10% for testing. In this setting, observations containing samples from both training and validation/test sets are discarded.

We adapt the DNN architecture for deep ranking regression algorithms based on the dataset type: for datasets with numerical features, we use fully-connected architectures with number of layers tuned w.r.t. best validation performance. For datasets with images as samples, we use the GoogleNet architecture (Szegedy et al., 2015) initialized with pre-trained weights on the ImageNet dataset. For the ICLR dataset with paper abstracts as samples, we use the Bidirectional Encoder Representations from Transformers (BERT) architecture with pre-trained weights on the Wikipedia dataset.



# Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> I. Yildiz, J. Dy, D. Erdogmus, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Deep Spectral Ranking”, AISTATS, San Diego, 2021

# Acknowledgements
Our work is supported by NIH (R01EY019474), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).
