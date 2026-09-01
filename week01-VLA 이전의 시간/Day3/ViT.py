from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch.nn.functional as F
import torch

# 사전학습 모델 다운로드
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 테스트 이미지
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 전처리 -> 추론
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(inputs)
print(inputs["pixel_values"].shape)

predicted_idx = outputs.logits.argmax(-1).item()
print(model.config.id2label[predicted_idx])

probs = F.softmax(outputs.logits, dim=-1)
top5 = probs[0].topk(5)
for p, idx in zip(top5.values, top5.indices):
    print(f"{p.item()*100:5.1f}% - {model.config.id2label[idx.item()]}")