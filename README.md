# DeepSpectralRanking
The code in this repository implements the algorithms and experiments in the following paper:  
> I. Yildiz, J. Dy, D. Erdogmus, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Deep Spectral Ranking”, AISTATS, San Diego, 2021

To begin with, install all package dependencies via `keras_gpu_env_3_6.yml`. The following command executes this in an Anaconda environment:
```
conda env create -f keras_gpu_env_3_6.yml
```

We implement five ranking regression algorithms that regress Plackett-Luce scores from ranking observations via Maximum Likelihood Estimation. 
- Deep Spectral Ranking (DSR)

# Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> I. Yildiz, J. Dy, D. Erdogmus, S. Ostmo, J. P. Campbell, M. F. Chiang, S. Ioannidis, “Deep Spectral Ranking”, AISTATS, San Diego, 2021

# Acknowledgements
Our work is supported by NIH (R01EY019474), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).
