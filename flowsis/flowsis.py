import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtdetrv2 import RTDetrV2

class FlowSIS(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rtdetr = RTDetrV2()
        
    def forward(self, image):
        rtdetr_outputs = self.rtdetr(image)
        img_features = rtdetr_outputs.encoder_last_hidden_state
        detections = 
        print()