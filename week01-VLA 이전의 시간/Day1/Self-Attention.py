# self-attention

import torch
import torch.nn.functional as F
from math import sqrt

def self_attention(x, W_q, W_k, W_v):
    """
    x : (batch, seq_len, d_model) -> 입력 토큰 (문장 1개, 토큰 4개, 각 토큰의 숫자 8개)
    W_q, W_k, W_v : (d_modle, d_k) -> Query, Key, Value 변환 행렬
    반환 : (batch, seq_len, d_k) -> attention이 적용된 새 표현
    """
    Q = x @ W_q  # (batch, seq_len, d_k)
    K = x @ W_k
    V = x @ W_v

    d_k = K.shape[-1]  # 벡터의 길이
    scores = Q @ K.transpose(-2, -1)  # Query, Key의 닮은 정도
    s_size = scores / sqrt(d_k)    # 점수 크기 조정
    weights = F.softmax(s_size, dim = -1) # 비율로 변환
    out = weights @ V    # 다른 토큰들의 Value를 가중 평균
    return out, weights

# 길이 4 시퀀스, 차원 8로 테스트
torch.manual_seed(0)
x = torch.randn(1, 4, 8)
W_q = torch.randn(8, 8)
W_k = torch.randn(8, 8)
W_v = torch.randn(8, 8)

out, weights = self_attention(x, W_q, W_k, W_v)
print("weights : ")
print(weights)
print(weights.shape)

print("out : ")
print(out)
print(out.shape)