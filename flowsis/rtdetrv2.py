import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
from transformers.utils.generic import ModelOutput

class RTDetrV2(nn.Module):
    def __init__(self, model_id:str="PekingU/rtdetr_v2_r18vd"):
        super().__init__()

        self.processor = RTDetrImageProcessor.from_pretrained(model_id, cache_dir="flowsis/models")
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id, cache_dir="flowsis/models")
        
    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=False)
        return outputs
       
    def detect(self):
        pass
       
if __name__ == '__main__':
    image = Image.open('test.jpg')
    asdf = RTDetrV2()
    qwer = asdf(image)
    
    
'''
I have an RT-DETRv2 model in Huggingface. I want to fine-tune it using some videos. For now, assume there is one video for simplicity. I have a file that maps frame ids within the video to xyxy bounding boxes which represent ground truth objects to train on. 
'''
