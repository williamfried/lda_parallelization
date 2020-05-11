# Overview
A parallel implementation of Latent Dirichlet Allocation.

# Website
You can find the website at this URL:
https://lda-parallelization-cs205.github.io/lda_parallelization

All the source code for the website is located in this repo under the `docs` folder. To update the website, simply change these files and push to master.

There will be a README.md in that subdirectory where you can find further instructions.

# Access and IAM Roles
To enable everyone to work off the same master node / EC2 instance, as well as enable the team to see the logs, dashboard, spin up the new VMs, etc, we've decided to work off the same account using a set of IAM roles.

To SSH into the master node––aside from creating a new user and signing in through that user––you just have to create a key-pair under the EC2 dashboard. From there, we just have to add the public key into the `.ssh/authorized_keys` file.

# Submitting a Parallel MPI Job
To submit a job:
`awsbsub -n <NUMBER_OF_NODES> -cf <FILE>`

To monitor:
`watch awsbstat -d`

To check the output:
`awsbout -s <job_ID>#<nodenumber>`

To kill a job:
`awsbstat -s ALL`

As a starter on the master node, we've set up AWS' example `submit_mpi.sh` file, so you can just practice the above with this file.

The above is a watered-down set of instructions for what we need. For a more in-depth manual, check out these docs:
- https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials_03_batch_mpi.html
