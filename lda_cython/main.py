from cgs import LDA
import os.path
from mpi4py import MPI
import subprocess
docs_dir = "output"
dictionary_file = "word2num.csv"
docs_file = os.path.join(docs_dir,
                         "part-{:05}".format(MPI.COMM_WORLD.Get_rank()))
num_docs = int(subprocess.run(["wc", "-l", docs_file],
                              capture_output=True).stdout.split()[0])
num_tokens = int(subprocess.run(["wc", "-l", dictionary_file],
                                capture_output=True).stdout.split()[0])
lda_obj = LDA(docs_file, 10, 0.1, 0.1, num_tokens, num_docs)
lda_obj.sample(2)
