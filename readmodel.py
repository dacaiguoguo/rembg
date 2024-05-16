import torch

# 加载模型
model = torch.load('/Users/yanguosun/Developer/rembg/rmb14model.pth', map_location=torch.device('cpu'))
print(model)