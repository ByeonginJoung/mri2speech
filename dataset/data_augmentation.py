import torch
import torch.nn.functional as F
import numpy as np
import librosa

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x, video):
        for transform in self.transforms:
            x, video = transform(x, video)
        return x, video

# Gaussian Noise Transformation
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, x, video):
        noise = torch.randn_like(video) * self.std + self.mean
        return x, torch.clamp(video + noise, 0, 1)
