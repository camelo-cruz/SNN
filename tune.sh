#!/bin/bash
#SBATCH --job-name=snn_tuning            # Job name
#SBATCH --time=05-00:00
#SBATCH --output=logs/snn_tuning_%j.out  # Standard output log
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Create a directory for logs if it doesn't exist
mkdir -p logs

# Proxy settings (if required)
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

# Load necessary modules and activate environment
module load lang/Anaconda3/2020.11
source activate snn

python tune.py
