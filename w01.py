import os
import time
import numpy as np
import torch
import tqdm
from deep_privacy.utils import load_checkpoint, save_checkpoint
from deep_privacy.models.base_model import ProgressiveBaseModel

torch.manual_seed(0)
checkpointFile = input("Please enter the path of the checkpoint file to be loaded\n")
loadedCkpt = load_checkpoint(checkpointFile, load_best=False, map_location=None)
print("Discriminator Parameters: " + str(loadedCkpt["D"]["parameters"]))
print("Generator Parameters: " + str(loadedCkpt["G"]["parameters"]))

