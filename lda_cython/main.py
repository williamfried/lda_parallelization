from cgs import LDA
import os.path
from mpi4py import MPI
import subprocess
import numpy as np
import time

docs_dir = "output"
dictionary_file = "word2num.csv"

rank = MPI.COMM_WORLD.Get_rank()
# Assuming Spark naming convention
docs_file = os.path.join(docs_dir,
                         "part-{:05}".format(rank))
# Count number of lines
num_docs = int(subprocess.run(["wc", "-l", docs_file],
                              stdout=subprocess.PIPE).stdout.split()[0])
# Count number of lines
num_tokens = int(subprocess.run(["wc", "-l", dictionary_file],
                                stdout=subprocess.PIPE).stdout.split()[0])
lda_obj = LDA(docs_file, 10, 0.1, 0.1, num_tokens, num_docs)
for i in range(1):
    start = time.time()
    lda_obj.sample(1)
    end = time.time()
    coherence = lda_obj.get_topic_coherence()
    if rank == 0:
        print(end - start)
        print(np.average(coherence))
