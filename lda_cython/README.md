#Instructions for running
1. Make sure the dependencies are installed (see `requirements.txt`).
2. Run the spark job `spark-submit spark.py` to preprocess the sample data provided, `ap.txt`.
3. Build the cython code. Change the `setup.py` compile and linking flags as appropriate for your environment. Then run `python setup.py build_ext -i`. Now `cgs` and `setup` can be imported in python as modules.
4. Run the example code `main.py` using `mpiexec -n <num_mpi_nodes> python main.py`,

