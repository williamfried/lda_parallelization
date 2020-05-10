from cgs import LDA
import os.path
from mpi4py import MPI
import subprocess
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--docs", default="patents",
    help="Directory containing files named 'part-{:05}' where each line "
         "consists of integers (separated by non-digit characters), with the "
         "first representing the document index, and every subsequent pair "
         "representing a token id and the number of occurrences of that "
         "token in that document."
)
parser.add_argument("--results", default="results",
                    help="Directory to write results to.")
parser.add_argument(
    "--num_docs", default=32, type=int,
    help="Number of files to examine from --docs directory. Files 'part-00000' "
         "to f'part-{num_docs:05}' will be used."
)
parser.add_argument("--topics", default=16, type=int,
                    help="Number of topics")
parser.add_argument("--tokens", default=1281799, type=int,
                    help="Number of tokens to use. The tokens with "
                         "id less than num_tokens will be used.")
parser.add_argument("--threads", default=4, type=int,
                    help="Number of OpenMP threads to use.")
parser.add_argument("--iterations", default=128, type=int,
                    help="Number of iterations to sample.")
args = parser.parse_args()
docs_dir = args.docs
results_dir = args.results
total_docs_files = args.num_docs
num_topics = args.topics
num_tokens = args.tokens
num_threads = args.threads
num_iterations = args.iterations

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
# Assuming Spark naming convention
docs_list_index = (total_docs_files * rank // size,
                   total_docs_files * (rank + 1) // size)
docs_file_list = [os.path.join(docs_dir, "part-{:05}".format(index))
                  for index in range(*docs_list_index)]
# Count number of lines
num_docs = sum(int(subprocess.run(["wc", "-l", docs_file],
                              stdout=subprocess.PIPE).stdout.split()[0])
               for docs_file in docs_file_list)
# Count number of lines
# num_tokens = int(subprocess.run(["wc", "-l", dictionary_file],
#                                 capture_output=True).stdout.split()[0])
start_init = time.time()
lda_obj = LDA(docs_file_list, num_topics, 1 / num_topics, 1 / num_topics,
              num_tokens, num_docs,
              seed=0,
              num_threads=num_threads, shuffle_words=True)
if rank == 0:
    print("Initialization time (s):", time.time() - start_init)
coherence_list = list()
for i in range(num_iterations):
    # The purpose of the barrier is just to get accurate timings
    MPI.COMM_WORLD.Barrier()
    start = time.time()
    lda_obj.sample()
    end = time.time()
    coherence = lda_obj.get_topic_coherence()
    if rank == 0:
        print(f"Time for iteration {i} (s):", end - start)
        coherence_list.append(coherence)
top_tokens = lda_obj.get_top_tokens(50)
if rank == 0:
    np.save(os.path.join(results_dir, "coherence"), coherence_list)
    np.save(os.path.join(results_dir, "top_tokens"), top_tokens)
    np.save(os.path.join(results_dir, "token_map"), lda_obj.get_inverse_permutation())
