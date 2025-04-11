#!/bin/bash
#SBATCH --job-name=snn_training          
#SBATCH --time=00-10:00                
#SBATCH --output=logs/chl_training_%j.out  # Standard output log
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Ensure logs directory exists
mkdir -p logs

# Proxy settings (if required)
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128

# Load necessary modules
module load lang/Anaconda3/2020.11

# Activate the conda environment
source activate snn

# Run the Python script
python ContrastiveNetwork.py || echo "Python script failed!"
