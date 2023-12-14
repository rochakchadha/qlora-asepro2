#!/bin/bash
# Install Python packages from requirements.txt
echo "Installing Python packages from requirements.txt..."
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip3 install -r requirements.txt

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs

echo "Cloning the Mistral-7B-v0.1 repository..."
cd base_model
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1

wandb login