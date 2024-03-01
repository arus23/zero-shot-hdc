#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here:
  # conda install -c nvidia cudatoolkit nvidia cuda-nvcc pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  
  pip install -r /netscratch/shanbhag/zero-shot-diffusion-classifier/requirements.txt
  pip uninstall transformer-engine
  sed -i 's/from torchvision.models.utils import load_state_dict_from_url/from torch.hub import load_state_dict_from_url/g' /usr/local/lib/python3.10/dist-packages/robustness/imagenet_models/vgg.py
  sed -i 's/torchvision.models.utils import load_state_dict_from_url/torch.hub import load_state_dict_from_url/g' /usr/local/lib/python3.10/dist-packages/robustness/imagenet_models/alexnet.py
  sed -i 's/torchvision.models.utils import load_state_dict_from_url/torch.hub import load_state_dict_from_url/g' /usr/local/lib/python3.10/dist-packages/robustness/imagenet_models/squeezenet.py
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi

# This runs your wrapped command
"$@"