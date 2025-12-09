#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --account=l_iui_uelschen_pva_ws25
#SBATCH --partition=compute
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --time=00:05:00
#SBATCH --output=cuda_output_%j.log

# Load CUDA module
export PATH=/opt/ohpc/pub/easybuild/software/CUDA/11.8.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/ohpc/pub/easybuild/software/CUDA/11.8.0/lib64:$LD_LIBRARY_PATH

# Run the program
cd ~/cuda-projects/CUDA/build
./untitled4
