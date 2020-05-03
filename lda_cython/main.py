from cgs import LDA
import os.path
from mpi4py import MPI
import subprocess
import numpy as np

docs_dir = "output"
dictionary_file = "word2num.csv"

rank = MPI.COMM_WORLD.Get_rank()
docs_file = os.path.join(docs_dir,
                         "part-{:05}".format(rank))
num_docs = int(subprocess.run(["wc", "-l", docs_file],
                              capture_output=True).stdout.split()[0])
num_tokens = int(subprocess.run(["wc", "-l", dictionary_file],
                                capture_output=True).stdout.split()[0])
lda_obj = LDA(docs_file, 10, 0.1, 0.1, num_tokens, num_docs)
for i in range(5):
    lda_obj.sample(1)
    coherence = lda_obj.get_topic_coherence()
    if rank == 0:
        print(np.average(coherence))
