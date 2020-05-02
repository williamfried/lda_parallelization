# Overview
A parallel implementation of Latent Dirichlet Allocation.

# Website
You can find the website at this URL:
https://lda-parallelization-cs205.github.io/lda_parallelization

All the source code for the website is located in this repo under the `docs` folder. To update the website, simply change these files and push to master.

There will be a README.md in that subdirectory where you can find further instructions.

# Setting up the Parallel Cluster
For the parallel implementation, we've set up an AWS Batch cluster, using parallel cluster. We orchestrated this
locally on one of our computers, which is a MacOS Catalina, version 10.15.3.

**NOTE**: you won't have to repeat the steps of this section unless we want to create a new cluster with new configurations.

1. In your terminal, set up a virtual environment using pip3: `virtualenv -p /usr/bin/python3.8 cs205env`

2. Activate the virtual environment: `source cs205env/bin/activate`

2. Install the parallel cluster client:
`pip3 install aws-parallelcluster --upgrade --user`

2. Check to ensure that parallel cluster installed correctly:
`pcluster version`

3. Install the AWS cli client
`pip3 install awscli`

4. Configure your AWS cluster: `aws configure`. For this step, use the credentials for IAM users within the CS205 project group.


5. Now, we'll configure the cluster: `pcluster configure`. We picked us-east-2 for the region ID, and used awsbatch for the scheduler. In terms of size for our initial cluster, we picked 2 vcpus minimum and 10 max. We may need to adjust these values later. We used a c5.2xlarge instance type. we also allowed automated VPC creation.

6. Now create the cluster: `pcluster create <cluster name... in our case we used cs205>`

7. You should now be able to SSH into the master node from your local machine: `pcluster ssh cs205 -i /path/to/keyfile.pem`. You should also be able to ssh using the public DNS. This works totally fine, but you should ensure that you log in with user `ec2-user`.


If you run into issues or require more than these steps, you can check out the more in depth set of manuals through Amazon's documentation, which we used in doing this setup:
- https://docs.aws.amazon.com/parallelcluster/latest/ug/install.html#install-tool-venv
- https://docs.aws.amazon.com/parallelcluster/latest/ug/getting-started-configuring-parallelcluster.html
- https://docs.aws.amazon.com/parallelcluster/latest/ug/tutorials_03_batch_mpi.html


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

^^^ This section especially is something I'm just writing up for the purposes of communicating an easy set-up / making it easier to just reference one set of docs / keep everything in one place as we develop. We can delete this later when we go in for the final submission.
