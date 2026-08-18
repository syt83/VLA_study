# VLA Study

**Vision-Language-Action (VLA) 모델을 처음부터 이해하고 직접 구현·실험해보는 학습 기록**

> 이 저장소는 VLA를 공부하면서 배운 개념, 논문, 수식, 코드, 시뮬레이션 결과를 주차별로 기록하는 개인 학습 저장소다.

---

## VLA

최근 로봇 분야에서는 단순히 센서 데이터를 받아 움직이는 것을 넘어,

> **"로봇이 세상을 보고(Vision), 사람의 말을 이해하고(Language), 실제 행동(Action)을 수행하게 만드는 것"**

이 중요해지고 있다.

VLA(Vision-Language-Action)는 이러한 흐름을 대표하는 모델 구조다.

하지만 처음부터 VLA 논문을 읽으려고 하니까 문제가 있었다.

Transformer, Vision Transformer, CLIP부터 시작해서
Behavior Cloning, Diffusion Policy, Robot Policy를 거쳐
최종적으로 VLA가 등장하는 과정을 제대로 이해하지 못한 상태에서는

> "이 논문은 기존 모델에서 뭐가 달라진 거지?"

라는 생각이 계속 들었다.

그래서 이 저장소에서는 **최신 VLA 모델부터 거꾸로 공부하지 않고, VLA가 등장하기까지의 흐름을 처음부터 따라가본다.**

---

# Study Roadmap

전체적인 흐름은 다음과 같이 잡았다.

```text
Transformer
     │
     ▼
Vision Transformer (ViT)
     │
     ▼
Vision-Language Model (CLIP)
     │
     ▼
Decision Transformer
     │
     ▼
Behavior Cloning
     │
     ▼
Robot Policy
     │
     ├──────────────┐
     ▼              ▼
ACT          Diffusion Policy
     │              │
     └──────┬───────┘
            ▼
       VLA 등장
            │
     ┌──────┴──────┐
     ▼             ▼
   RT-2          PaLM-E
     │
     ▼
Open X-Embodiment
     │
     ▼
Octo / OpenVLA / RDT
     │
     ▼
π₀ / π₀.₅
     │
     ▼
Gemini Robotics / GR00T / SmolVLA
```

---

# Weekly Study

## Week 01 — VLA 이전의 시간

### Transformer · ViT · CLIP · Decision Transformer

VLA를 이해하기 위한 기본기를 만든다.

### 공부할 것

* Transformer
* Self-Attention
* Multi-Head Attention
* Positional Encoding
* Vision Transformer
* CLIP
* Decision Transformer

### 직접 해볼 것

* [x] Scaled Dot-Product Attention 구현
* [ ] Multi-Head Attention 구현
* [ ] 간단한 ViT 구조 구현
* [ ] CLIP inference
* [ ] Decision Transformer 구조 분석

### 핵심 질문

> Transformer가 어떻게 이미지와 언어를 처리할 수 있고, 더 나아가 행동까지 처리할 수 있을까?

[→ Week 01](./week01-transformer/)

---

## Week 02 — 모방학습

### BC-Z · Gato · RT-1

로봇에게 행동을 어떻게 학습시킬 수 있는지 공부한다.

### 공부할 것

* Behavior Cloning
* Demonstration Dataset
* Robot Policy
* BC-Z
* Gato
* RT-1

### 직접 해볼 것

* [ ] 간단한 Behavior Cloning 구현
* [ ] Observation → Action 구조 구현
* [ ] RT-1 구조 분석

### 핵심 질문

> 사람이 로봇을 조작해서 만든 데이터를 이용해 로봇이 행동을 배울 수 있을까?

[→ Week 02](./week02-imitation-learning/)

---

## Week 03 — 정밀 모방학습의 두 길

### ACT · Diffusion Policy · Mobile ALOHA

로봇의 행동을 더 정밀하게 생성하는 방법을 공부한다.

### 공부할 것

* Action Chunking
* ACT
* Diffusion Policy
* Mobile ALOHA
* Teleoperation
* Imitation Learning

### 직접 해볼 것

* [ ] ACT 실행
* [ ] Diffusion Policy 실행
* [ ] PushT 시뮬레이션
* [ ] ACT와 Diffusion Policy 결과 비교

### 핵심 질문

> 로봇의 행동을 하나씩 예측하는 것보다 더 좋은 방법은 무엇일까?

[→ Week 03](./week03-precision-imitation/)

---

## Week 04 — VLA가 정식으로 등장하다

### PaLM-E · RT-2 · RoboCat

Vision과 Language를 Robot Action으로 연결하는 VLA의 등장을 공부한다.

### 공부할 것

* PaLM-E
* RT-2
* RoboCat
* Vision-Language-Action
* Multimodal Learning
* Robot Action Representation

### 핵심 질문

> 기존 Vision-Language Model이 어떻게 로봇 행동을 수행할 수 있게 되었을까?

[→ Week 04](./week04-vla-emergence/)

---

## Week 05 — 오픈소스가 따라잡다

### Open X-Embodiment · Octo · OpenVLA · RDT-1B

여러 로봇과 여러 Task를 하나의 모델로 처리하려는 흐름을 공부한다.

### 공부할 것

* Open X-Embodiment
* Octo
* OpenVLA
* RDT-1B
* Multi-Embodiment Learning
* Generalist Robot Policy

### 직접 해볼 것

* [ ] OpenVLA inference
* [ ] LoRA fine-tuning
* [ ] Open-X 또는 DROID Dataset 살펴보기
* [ ] SimplerEnv에서 평가

### 핵심 질문

> 서로 다른 로봇에서 수집된 데이터를 이용하면 새로운 로봇과 Task에도 일반화할 수 있을까?

[→ Week 05](./week05-open-source-vla/)

---

## Week 06 — π 시리즈와 산업화의 시작

### π₀ · π₀-FAST · π₀.₅

최근 VLA가 실제 로봇 환경에서 어떤 방향으로 발전하고 있는지 공부한다.

### 공부할 것

* π₀
* π₀-FAST
* π₀.₅
* Action Representation
* Flow Matching
* Generalization

### 직접 해볼 것

* [ ] π₀ 구조 분석
* [ ] π₀ inference
* [ ] Action generation 과정 분석

### 핵심 질문

> 최신 VLA는 이전 VLA와 비교해서 행동을 어떻게 다르게 생성할까?

[→ Week 06](./week06-pi-series/)

---

## Week 07 — 휴머노이드와 최전선

### Helix · Gemini Robotics · GR00T N1 · SmolVLA

현재 VLA가 어디까지 발전했는지 살펴본다.

### 공부할 것

* Helix
* Gemini Robotics
* GR00T N1
* SmolVLA
* Humanoid Robot
* Generalization
* Real-world Deployment

### 직접 해볼 것

* [ ] SmolVLA 실행
* [ ] VLA inference
* [ ] 시뮬레이션 환경에서 행동 확인
* [ ] 최신 VLA 구조 비교

### 핵심 질문

> 현재 VLA가 실제 로봇과 휴머노이드에서 해결해야 하는 가장 큰 문제는 무엇일까?

[→ Week 07](./week07-humanoid-frontier/)

---

# 🧪 실습 환경

이번 공부에서는 **실물 로봇을 사용하지 않고 노트북에서 시뮬레이션과 공개 Dataset을 이용한다.**

실제 로봇 하드웨어 문제보다 VLA의 구조와 알고리즘 자체에 집중하는 것을 목표로 한다.

## 주요 환경

```text
Python
PyTorch
Hugging Face
LeRobot
CUDA
Git / GitHub
```

## Simulation / Dataset

```text
PushT
ALOHA Simulation
SimplerEnv
Open X-Embodiment
DROID
```

---

# 💻 실습 원칙

단순히 기존 코드를 실행하는 것에서 끝내지 않는다.

가능하면 다음 순서로 공부한다.

```text
개념 정리
   ↓
수식 이해
   ↓
간단한 코드 직접 구현
   ↓
기존 모델 실행
   ↓
시뮬레이션
   ↓
결과 분석
   ↓
VLA와 연결
```

특히 **수식과 코드가 어떻게 대응되는지**를 중요하게 기록한다.

예를 들어 Attention을 공부할 때:

[
Attention(Q,K,V)
================

softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
]

를 단순히 외우는 게 아니라,

```python
scores = Q @ K.T
scores = scores / math.sqrt(d_k)
attention_weights = torch.softmax(scores, dim=-1)
output = attention_weights @ V
```

처럼 실제 코드와 연결해서 이해한다.

---

# 📝 Weekly Study Format

각 주차의 README는 가능한 한 동일한 형식으로 작성한다.

```text
1. 학습 목표
      ↓
2. 배경 지식
      ↓
3. 핵심 개념
      ↓
4. 핵심 수식
      ↓
5. 모델 / 알고리즘 구조
      ↓
6. 논문 분석
      ↓
7. 코드 구현
      ↓
8. 시뮬레이션 / 실험
      ↓
9. 결과 분석
      ↓
10. 내가 이해한 것
      ↓
11. 아직 이해하지 못한 것
      ↓
12. VLA와 연결
      ↓
13. 모델 발전 과정에서의 위치
      ↓
14. 한계와 개선 방향
      ↓
15. 이번 주 정리
      ↓
16. 다음 주
```

---

# 🔬 실험 기록

단순히 모델을 실행한 결과만 기록하지 않는다.

가능하면 다음 내용을 함께 기록한다.

| 항목             | 내용 |
| -------------- | -- |
| Model          |    |
| Dataset        |    |
| Environment    |    |
| Task           |    |
| Training Steps |    |
| Batch Size     |    |
| Learning Rate  |    |
| Success Rate   |    |
| Inference Time |    |
| 결과             |    |
| 문제점            |    |

그리고 가능하면 실험 결과를 이미지나 GIF로 저장한다.

```text
assets/
├── experiment_01.gif
├── experiment_02.gif
└── result.png
```

---

# 🧠 내가 중요하게 볼 것

VLA 모델을 공부할 때 단순히 모델의 이름과 성능을 외우지 않는다.

각 모델마다 다음 질문을 계속 던진다.

### 1. 무엇을 입력으로 받는가?

```text
Image?
Language?
Robot State?
Previous Action?
```

### 2. 어떤 구조를 사용하는가?

```text
CNN?
ViT?
Transformer?
Diffusion?
Flow Matching?
```

### 3. Action은 어떻게 표현하는가?

```text
Discrete Token?
Continuous Action?
Action Chunk?
Diffusion?
Flow?
```

### 4. 어떤 데이터로 학습하는가?

```text
Human Demonstration?
Robot Dataset?
Internet Data?
Multi-Robot Dataset?
```

### 5. 무엇이 이전 모델보다 좋아졌는가?

### 6. 어떤 문제가 아직 남아 있는가?

### 7. Generalization은 어떻게 되는가?

---

# 🌎 VLA Generalization

이번 공부에서 특히 관심 있게 볼 주제는 **Generalization**이다.

VLA는 학습한 환경과 다른 상황에서도 로봇이 제대로 행동할 수 있어야 한다.

예를 들어:

```text
Training
────────────────────
Robot A
Task A
Environment A
Object A
      ↓
     VLA
────────────────────

Testing
────────────────────
Robot B
Task B
Environment B
Object B
      ↓
     ?
────────────────────
```

따라서 각 논문을 공부할 때마다

> **"이 모델은 새로운 환경, 새로운 물체, 새로운 Task, 새로운 Robot에 얼마나 잘 일반화되는가?"**

를 확인한다.

---


# 📈 최종 목표

이 공부가 끝났을 때 단순히

> "VLA가 무엇인지 안다."

에서 끝내지 않는다.

최종적으로는 다음 흐름을 설명할 수 있는 것을 목표로 한다.

```text
Transformer
      ↓
Attention
      ↓
ViT
      ↓
CLIP
      ↓
Behavior Cloning
      ↓
Robot Policy
      ↓
ACT / Diffusion Policy
      ↓
RT-1
      ↓
PaLM-E / RT-2
      ↓
OpenVLA / Octo
      ↓
π₀
      ↓
최신 VLA
```

그리고 각 단계에서

> **왜 다음 기술이 필요했는지**

를 설명할 수 있도록 한다.

---

# 🚀 Final Goal

최종적으로는 직접 간단한 VLA 구조를 구현하고,

```text
Image
   +
Language Instruction
   ↓
Vision Encoder
   +
Language Encoder
   ↓
Multimodal Representation
   ↓
Policy
   ↓
Action
   ↓
Simulation Robot
```

의 전체 과정을 이해하는 것을 목표로 한다.

그리고 마지막에는 지금까지 공부한 모델들을 비교하면서

> **"VLA가 앞으로 해결해야 할 문제는 무엇인가?"**

에 대한 나만의 생각을 정리한다.

---

# 📌 Study Rule

> **모르는 것을 그냥 넘어가지 않는다.**

논문에서 이해되지 않는 수식이 나오면 수식을 쪼개서 공부하고,
코드가 이해되지 않으면 작은 코드로 다시 구현하고,
모델 구조가 이해되지 않으면 직접 그림을 그려본다.

그리고 단순히 **"무엇을 배웠는가"**보다

> **"왜 이렇게 만들었는가?"**

를 이해하는 것을 가장 중요하게 생각한다.

---

## Current Progress

* [ ] Week 01 — Transformer · ViT · CLIP · Decision Transformer
* [ ] Week 02 — BC-Z · Gato · RT-1
* [ ] Week 03 — ACT · Diffusion Policy · Mobile ALOHA
* [ ] Week 04 — PaLM-E · RT-2 · RoboCat
* [ ] Week 05 — Open X-Embodiment · Octo · OpenVLA · RDT-1B
* [ ] Week 06 — π₀ · π₀-FAST · π₀.₅
* [ ] Week 07 — Helix · Gemini Robotics · GR00T N1 · SmolVLA

---

> **목표: VLA를 외우는 것이 아니라, VLA가 왜 지금의 구조가 되었는지 이해한다.**
