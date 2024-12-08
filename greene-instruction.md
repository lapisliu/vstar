## Running on Greene
```bash
#login to greene first.
#Then get an interactive node with a GPU
srun --cpus-per-task=4 --mem=80GB --gres=gpu:1 --time=04:00:00 --pty /bin/bash

#Go to your project folder. Here's my example.
cd $VAST
cd cv24/project/
```

### Get the singularity image
Download:
If it fails to download, you can install it following the next step.
```bash
#get the overlay file if not already
wget https://g-016e9b.00888.8540.data.globus.org/overlay-vstar.ext3
```
Install:
```bash
cp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
mv overlay-15GB-500K.ext3 overlay-vstar.ext3

#create a cache folder if not already.
#The cache folder should be either in the vast or scratch space, not in the home directory. In my case it's in the vast space.
mkdir cache

#The cache path before the colon here should be the cache folder path you just created. For example, mine's /vast/xl3893/cv24/project/cache
singularity exec --nv --bind /vast/xl3893/cv24/project/cache:$HOME/.cache --overlay overlay-vstar.ext3 /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif /bin/bash

wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
rm Miniforge3-Linux-x86_64.sh

vim /ext3/env.sh
```

Then, copy the following content to the env.sh file
```bash
#!/bin/bash

unset -f which

source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
```

Install the dependencies
```bash
#clone the repo if not already
source /ext3/env.sh

git clone https://github.com/lapisliu/vstar.git

#cd to vstar repo
cd vstar

#installation
conda create -n vstar python=3.10 -y
conda activate vstar
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export PYTHONPATH=$PYTHONPATH:path_to_vstar_repo
```

### Run the evaluation
```bash
source /ext3/env.sh
conda activate vstar
#run evaluation
git pull
python vstar_bench_eval.py --benchmark-folder bench/

#results are in the `eval_result.json` file.
```