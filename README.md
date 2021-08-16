# Overview
A parallel implementation of Latent Dirichlet Allocation.

# Website
You can find the website at this URL: https://lda-parallelization-cs205.github.io/lda_parallelization

My main contributions were:
1. serial implementation of collapsed Gibbs sampling (see [serial_cgs_driver.py](https://github.com/williamfried/lda_parallelization/blob/master/serial_cgs_driver.py) and [serial_cgs.py](https://github.com/williamfried/lda_parallelization/blob/master/serial_cgs.py)
2. parallelized implementation of collapsed Gibbs sampling using MPI (see [mpi_implementation_asynchronous.py](https://github.com/williamfried/lda_parallelization/blob/master/archive/mpi_implementation_asynchronous.py) and [mpi_implementation_synchronous.py](https://github.com/williamfried/lda_parallelization/blob/master/archive/mpi_implementation_synchronous.py))
3. Wrote entire description of algorithm for website.
