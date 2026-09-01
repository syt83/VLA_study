# Day 04 — ViT code review

# CLIP의 Zero-shot Classification을 실제로 실행

## 1. 후보 텍스트
```python
candidates = [
    "a photo of a black cat",
    "a photo of a brown cat",
    "a photo of a white cat",
    "a photo of a dog",
    "a photo of a robot arm",
    "a photo of a sandwich",
]
```

일반적인 이미지 분류기는 클래스가 모델 내부에 고정되어있다.
그러나 CLIP은 candidates = [...]에 사용자가 직접 후보를 넣어  
줄 수 있다.

>> 즉, 이번에 무엇과 무엇을 비교할지 사용자가 텍스트로 지정할 수 있다.

## 2. 이미지와 텍스트를 모델 입력 형태로 변환
```python
inputs = processor(
    text=candidates,
    images=image,
    return_tensors="pt",
    padding=True,
)
```
여기서 내부적으로 2가지 작업을 한다

### 1) 텍스트 처리
```
"a photo of a black cat"
      ↓
Tokenization
      ↓
input_ids           
```

### 2) 이미지 처리
```
PIL Image
   ↓
Resize
Normalize
Tensor 변환
   ↓
pixel_values
```

## 3. 추론 실행
```python
with torch.no_grad():
    outputs = model(**inputs)
```
실제 CLIP 모델에 이미지와 텍스트를 입력하여 결과를 얻는 부분이다.

여기서 torch.no_grad()는 gradient(기울기)를 계산하지 않도록 설정하는 것​이다.  
현재는 모델을 학습하는 것이 아니라 이미 학습된 모델로 결과를 확인하는 추론단계이므로,  
가중치 업데이트에 필요한 gradient 계산을 수행할 필요가 없다.

outputs = model(**inputs)
-> 모델 내부에 이미지 1개와 텍스트가 6개 가 각각 Encoder를 통과한다.

### 1) 이미지
입력된 이미지는 CLIP의 Vision Encoder인 ViT를 통과한다.
```
Image
  ↓
Vision Encoder
(ViT)
  ↓
Image Embedding
```
즉, 이미지가 숫자로 이루어진 하나의 Image Embedding​으로 변환된다.

### 2) 텍스트
6개의 후보 텍스트도 각각 CLIP의 Text Encoder를 통과한다.
```
Text 1 → Text Encoder → T₁
```

각 텍스트는 의미를 나타내는 Text Embedding​으로 변환된다.

### 3) 이미지와 텍스트의 유사도 비교
그다음 CLIP은 1개의 이미지 특징 벡터와 6개의 텍스트 특징 벡터를  
각각 비교하여 유사도를 계산한다.
```
Image ↔ Text 1 = score
Image ↔ Text 2 = score
Image ↔ Text 3 = score
...
```
즉, 이미지가 각 텍스트와 얼마나 의미적으로 유사한지 점수로 나타낸다.

이렇게 계산된 이미지와 텍스트 간의 유사도 점수가 이후 outputs.logits_per_image에 저장된다.

## 4. Softmax
```python
probs = outputs.logits_per_image.softmax(dim=-1)
```
여기서 softmax가 CLIP 사전학습에서 InfoNCE가 사용하는 Softmax와 같은 원리를 보여준다.

예를 들어 logits가
```
black cat = 2.0
brown cat  = 3.0
white cat = 1.0
dog       = -1.0
robot arm = -2.0
sandwich  = -3.0
```
이라면 Softmax를 적용해서
```
black cat = 25%
brown cat  = 68%
white cat = 6%
dog       = 0.8%
robot arm = 0.1%
sandwich  = 0.1%
```
처럼 텍스트들 사이에서의 확률 분포를 만든다.

### dim=-1을 쓰는 이유
현재 이미지가 1개, 텍스트 개수가 6개이므로 shape는 (1, 6)이다.  
여기서 dim=-1은 마지막 차원, 즉 후보 6개에 대해서 Softmax를 적용한다는 뜻이다.  
그래서 6개의 후보들의 확률의 합이 1이된다.

## 5. 결과 출력 
```python
for label, p in zip(candidates, probs[0]):
    print(f"{p.item()*100:5.1f}% — {label}")
```
이 두 줄은 후보 텍스트와 그에 해당하는 확률을 하나씩 짝지어서 출력하는 코드다.

여기서 probs[0]는 1개의 이미지의 결과를 가져온다.
```
probs.shape
(1, 6)

 ↓

probs[0]

(6,)
```
각 값은 candidates와 순서대로 대응된다.

- zip()은 두 리스트의 같은 위치에 있는 값끼리 묶어주는 함수이다.  
  ex) ("a photo of a black cat", 0.10)

- for label, p in zip(candidates, probs[0]):는 zip으로 묶인 값들을  
  하나씩 꺼내서 첫 번째 값은 label에, 두 번째 값은 p에 넣는다.
  ex) label = "a photo of a black cat"
      p = 0.10

- print(f"{p.item()*100:5.1f}% — {label}")
  p.item에서 p는 pytorch의 tensor 값이다.
  ex) p가 tensor(0.7000)일 때, p.item()은 0.7을 반환한다.

- * 100은 확률에 100을 곱하여 퍼센트로 바꾼다.

전체적인 구조는 
```
후보 텍스트 하나
        +
그 텍스트의 확률 하나
        ↓
짝지어서 하나씩 반복하고


print(f"{p.item()*100:5.1f}% — {label}")는

확률을 % 형태로 출력
        +
후보 텍스트 출력한다.
```

>> 후보 텍스트 6개와 각 후보의 확률 6개를 순서대로 하나씩 묶어서, 확률과 텍스트를 출력한다.

## 6. 결과
```
  1.4% — a photo of a black cat
 87.4% — a photo of a brown cat
  0.7% — a photo of a white cat
  0.6% — a photo of a dog
  1.7% — a photo of a robot arm
  8.2% — a photo of a sandwich
```
이미지 속 고양이를 갈색 고양이로 판단했으며, 가장 높은 확률인 **87.4%**를 보였다.

이미 사적 학습된 CLIP 모델이 갈색 고양이를 분류하기 위해 별도의 분류기를 새로  
학습하지 않고 잘 판단하는 것을 볼 수 있다.

즉, 추가적인 분류기 학습 없이 Zero-shot Classification을 수행하였다.


