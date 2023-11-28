# Unsupervised Image Segmentation With Mixture Models And Markov Random Fields
The code was used to obtain the results in our manuscript "A guide to unsupervised image segmentation of mCT-scanned cellular metal materials with mixture modelling and Markov random fields" by Branislav Panić, Matej Borovinšek, Matej Vesenjak, Simon Oman and Marko Nagode, which is currently under revision in Materials&Design.

# Python Package Requirements
- numba with cuda support
- numpy
- scikit-image
- tqdm
- yaml

Install them with pip!
# R Package Requirements
- rebmix
- imager
- yaml
- argparser

Install them inside running R console with "install.packages" function.

# Running main.py python script

Prerequisitions: It is best to put images in  "images/" directory where source code is located and create "labels/" directory for storing the segmentated images!

Following command line arguments can be supplied to the script:

- "--cwd": Current working directory. Supply with full path. Defaulting to os.getcwd().
- "--images-dir": Images directory. Supply with relative path to cwd. Defaulting to "images/". 
- "--pdf": Probability density function of mixture model. Supply with one of "normal", "lognormal", "gamma", "Weibull" or "Gumbel". Defaulting to "normal"
- "--cmax": Maximum possible number of components in mixture model for model selection. Supply with integer value. Defaulting to 64.
- "--cmin": Minimum possible number of components in mixture model for model selection. Supply with integer value. Defaulting to 1.
- "--target-pixel-number": Target pixel number that porous structure should contain. Supply with integer value. This argument does not have default value and is recommended. Please read paper for more information.
- "--merge-clusters-rounding": Rounding when calculating true/false clusters. Supply with one of "nearest", "upper" or "lower". Defaulting to "nearest".
- "--beta": Beta parameter for Markov random fields. Supply with positive float. Defaults to 1.
- "--nbr-of-icm-iters": Number of iterations of ICM algorithm. Currently, the ICM is not fully parallelized with CUDA so set this sparingly. Defaulting to 1.
- "--save-dir": Segmentation directory. Supply with relative path to cwd. Defaulting to "labels/"


# Example running script:

python main.py --target-pixel-number 1999231
