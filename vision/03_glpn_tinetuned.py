#######################
#######inference#######
#######################

from transformers import AutoImageProcessor, GLPNForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

### data load ###
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

### model load ###
model_name = 'vinvino02/glpn-kitti'
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = GLPNForDepthEstimation.from_pretrained(model_name)

inputs = image_processor(images=image, return_tensors="pt")

### inference ###
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

### post processing ###
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

#######################
######fine-tuning######
#######################

import torch
import h5py
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from transformers import GLPNForDepthEstimation, GLPNImageProcessor
import torch.nn.functional as F

# !wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
device = torch.device("cuda")

### data load ###
with h5py.File("nyu_depth_v2_labeled.mat", "r") as file:
    # images = torch.tensor(file['images'][()]).permute(0, 2, 3, 1)  # Convert to PyTorch format
    images = torch.tensor(file['images'][()])
    depths = torch.tensor(file['depths'][()]).unsqueeze(1)  # Convert to PyTorch format
dataset = TensorDataset(images.float(), depths.float())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

### model load ###
model_name = "vinvino02/glpn-kitti"
image_processor = GLPNImageProcessor.from_pretrained(model_name)
model = GLPNForDepthEstimation.from_pretrained(model_name)
model = model.to(device)

### train ###
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(2):  
    for i, (images, depths) in enumerate(dataloader):
        images = images.to(device)
        depths = depths.to(device)
        inputs = image_processor(images=images.to(device), return_tensors="pt").to(device)

        outputs = model(**inputs)
        loss = loss_fn(outputs.predicted_depth, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/10], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")