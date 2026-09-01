from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# CLIP 프로세서
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# CLIP 모델
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 이미지 다운로드
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 같은 이미지, 임의의 후보 텍스트
candidates = [
    "a photo of a black cat",
    "a photo of a brown cat",
    "a photo of a white cat",
    "a photo of a dog",
    "a photo of a robot arm",
    "a photo of a sandwich",
]

inputs = processor(
    text=candidates,
    images=image,
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    outputs = model(**inputs)

# logits_per_image: (1, len(candidates))
probs = outputs.logits_per_image.softmax(dim=-1)
for label, p in zip(candidates, probs[0]):
    print(f"{p.item()*100:5.1f}% — {label}")
