# Instructions for running
1. Make sure the dependencies are installed (see `requirements.txt`).
2. Preprocess the documents.
3. Build the cython code. Change the `setup.py` compile and linking flags as appropriate for your environment. Then run `python setup.py build_ext -i`. Now `cgs` and `utils` can be imported in Python as modules.
4. Run the example code `main.py` using `mpiexec -n <num_mpi_nodes> python main.py <options>`. Run `python main.py -h` for a description of the options.

