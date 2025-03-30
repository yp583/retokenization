#!/bin/bash
#SBATCH --job-name=test          # Job name
#SBATCH --output=test.txt     # Output log file
#SBATCH --error=test_error.txt       # Error log file
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=16             # Number of CPU cores per task
#SBATCH --mem=64G                      # Memory allocation
#SBATCH --gres=gpu:a100:1              # Request 1 A100 GPU
#SBATCH --time=04:00:00                # Max execution time (hh:mm:ss)
#SBATCH --partition=standard           # Choose appropriate partition (e.g., standard, gpu, largemem)
#SBATCH --mail-type=ALL                # Get email notifications
#SBATCH --mail-user=your_email@umich.edu  # Replace with your UMich email

# Load required modules
module load python/3.10 cuda/12.1 gcc/11.3 pytorch/2.1.0 

# Activate virtual environment (if applicable)
source ~/envs/llama3/bin/activate

# Run LLAMA3 script
python run_llama3.py --model llama3-8b --input "example_prompt.txt" --output "output.txt"
