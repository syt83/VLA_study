# Day 03 — ViT code review

# 사전학습된 ViT-Base 모델을 이용한 이미지 분류 실습

## 1. 모델 불러오기

### ViTImageProcessor
이미지를 ViT가 입력받을 수 있는 형태로 변환

```
원본 이미지
    ↓
Resize
    ↓
224 × 224
    ↓
Normalize
    ↓
Tensor
```

### ViTForImageClassification
분류 헤드까지 포함된 ViT

```
이미지
  ↓
Patch 분할
  ↓
Patch Embedding
  ↓
ViT Encoder
  ↓
대표 Feature
  ↓
Classification Head
  ↓
1000개 클래스
```

---

## 2. 전처리

```python
inputs = processor(
    images=image,
    return_tensors="pt"
)

결과 : 
torch.Size([1, 3, 224, 224])
```
즉, 현재 이 단계에선 패치 토큰이 만들어진 상태가 아니다.

---

## 3. ViT 추론

```python
with torch.no_grad():
    outputs = model(**inputs)
```
실제 내부에서 발생하는 일은 다음과 같다

```
pixel_values
[1, 3, 224, 224]

        ↓

Patch 분할

        ↓

Patch Embedding

        ↓

Visual Tokens

        ↓

ViT Transformer Encoder

        ↓

Classification Head

        ↓

logits
```

---

## 4. logits
logits는 모델이 각 클래스에 대해 낸 원본 점수이며, 가장 높은 점수를 가진 클래스가 모델의 최종 예측이 된다.

ex)
```
outputs.logits

인덱스:   0      1      2      3
클래스:  고양이  강아지  자동차  비행기
점수:    2.1    8.5    0.3    1.2
```

코드에서
```python
predicted_idx = outputs.logits.argmax(-1).item()

실행한 후 각 부분은 : 
outputs.logits
→ 모든 클래스의 점수

.argmax(-1)
→ 가장 큰 점수가 있는 위치(인덱스)를 찾음

.item()
→ PyTorch Tensor 값을 일반 Python 숫자로 변환

이다.
```

따라서 위 예시에서는
```
outputs.logits = [2.1, 8.5, 0.3, 1.2]

outputs.logits.argmax(-1)
→ 1

predicted_idx
→ 1
```

그리고 
```python
model.config.id2label[predicted_idx] 을 실행하면

1 → "dog"
처럼 인덱스 번호를 실제 클래스 이름으로 변환할 수 있다.
```

실제 작성한 코드를 실행해보면
```
Egyptian cat
```

여기서 logits는 확률이 아니기 때문에, 확률로 본다면 Softmax를 적용해야 한다.

## 5. Softmax

```python
probs = F.softmax(
    outputs.logits,
    dim=-1
)
```

logits를 확률로 변환한다.
```
logits
[8.2, 2.1, 0.3]

        ↓

Softmax

        ↓

[0.997, 0.002, 0.001]
```

여기서 logits의 모든 확률을 더하면 1이된다.

---

## 6. Top-5 출력
```python
top5 = probs[0].topk(5)
```
가장 높은 확률 5개를 가져온다.  

이를 출력한 결과는
```
 93.7% - Egyptian cat
  3.8% - tabby, tabby cat
  1.4% - tiger cat
  0.3% - lynx, catamount
  0.1% - Siamese cat, Siamese
```
이다.