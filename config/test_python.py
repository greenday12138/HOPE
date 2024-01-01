import glob
import logging
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path

# import cv2
import numpy as np
# import torch
# import torchvision
import yaml

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
save_txt = True
test_path='/home/xiayuyang/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking/'
save_dir = Path(increment_path(Path(test_path)))
print(save_dir)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
save_dir = Path(increment_path(Path(test_path)))
print(save_dir)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
save_dir = Path(increment_path(Path(test_path)))
print(save_dir)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
save_dir = Path(increment_path(Path(test_path)))
print(save_dir)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
save_dir = Path(increment_path(Path(test_path)))
print(save_dir)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 
save_dir = Path(increment_path(Path(test_path)))
print(save_dir)
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 