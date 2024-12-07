## Running on Greene
```bash
#login to greene first.
#Then get an interactive node with a GPU
srun --cpus-per-task=4 --mem=80GB --gres=gpu:1 --time=04:00:00 --pty /bin/bash

#Go to your project folder. Here's my example.
cd $VAST
cd cv24/project/

#get the overlay file if not already
wget https://g-016e9b.00888.8540.data.globus.org/overlay-vstar.ext3

#create a cache folder if not already
mkdir cache

#The cache path before the colon here should be the cache folder path you just created. For example, mine's /vast/xl3893/cv24/project/cache
singularity exec --nv --bind /vast/xl3893/cv24/project/cache:$HOME/.cache --overlay overlay-vstar.ext3 /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash
source /ext3/env.sh
conda activate vstar

#clone the repo if not already
git clone https://github.com/lapisliu/vstar.git

#cd to vstar repo
cd vstar
#run evaluation
git pull
python vstar_bench_eval.py --benchmark-folder bench/

#results are in the `eval_result.json` file.
```