import numpy as np
import matplotlib.pyplot as plt

# (0,0), (0,1), (1,0), (1,1) 근처의 점들로 구성
X = np.array([
    [0.1, 0.2], [0.2, 0.1], [0.1, 0.1], # (0,0) 구간 -> 파랑
    [0.8, 0.9], [0.9, 0.8], [0.9, 0.92], # (1,1) 구간 -> 파랑 (XOR는 같으면 0)
    [0.1, 0.9], [0.2, 0.8], [0.1, 0.8], # (0,1) 구간 -> 빨강
    [0.9, 0.1], [0.8, 0.2], [0.8, 0.1]  # (1,0) 구간 -> 빨강
])

# XOR 논리: 두 값이 서로 다르면 빨강(1), 같으면 파랑(0)
색깔 = []
라벨 = []
for x in X:
    if (x[0] < 0.5 and x[1] < 0.5) or (x[0] > 0.5 and x[1] > 0.5):
        색깔.append('blue') # (0,0) 이나 (1,1) 근처
        라벨.append(0)
    else:
        색깔.append('red')  # (0,1) 이나 (1,0) 근처
        라벨.append(1)

라벨 = np.array(라벨) # 리스트를 넘파이 배열로 변환

# W(1) 행렬: [[-2, 2], [-2, 2]]
W1 = np.array([
    [-2.0,  2.0],
    [-2.0,  2.0]
])

# B(1) 벡터: [3, -1]
b1 = np.array([3.0, -1.0])

print("--- 적용된 가중치(W) ---")
print(W1)
print("\n--- 적용된 바이어스(b) ---")
print(b1)

# (1) 선형 변환: Z = X * W + b
Z = np.dot(X, W1) + b1

# (2) 활성화 함수: 시그모이드 (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

H = sigmoid(Z)

# 마지막 레이어 가중치
w31, w32 = 1.0, 1.0

# 점수 계산: n1 + n2 (가중치가 1, 1이니까 그냥 더하면 됨)
scores = H[:, 0] * w31 + H[:, 1] * w32

# 파란 팀(0)과 빨간 팀(1) 점수 나누기
blue_scores = scores[라벨 == 0]
red_scores  = scores[라벨 == 1]

# 경계선 찾기: 파란 팀의 최대값과 빨간 팀의 최소값의 '중간'
# 파란 팀은 (0,0)이나 (1,1) 쪽이라 값이 작고, 빨간 팀은 가운데라 값이 큼
max_blue = np.max(blue_scores) # 파란 팀 중 제일 높게 올라온 놈
min_red  = np.min(red_scores)  # 빨간 팀 중 제일 낮게 내려온 놈

# 둘 사이의 딱 중간 지점 (Midpoint)
optimal_threshold = (max_blue + min_red) / 2

# 바이어스는 임계값의 반대 부호 (수식: n1 + n2 + b = 0  ->  b = -(n1+n2))
auto_b3 = -optimal_threshold

print(f"파란 팀 최대 점수: {max_blue:.4f}")
print(f"빨간 팀 최소 점수: {min_red:.4f}")
print(f"자동으로 계산된 최적 바이어스(b3): {auto_b3:.4f}")

# 시각화
plt.figure(figsize=(10, 5))

# 왼쪽: 입력 공간 (Input)
plt.subplot(1, 2, 1)
plt.title("1. Input Layer (x1, x2)")
plt.scatter(X[:, 0], X[:, 1], c=색깔, s=100, alpha=0.7)
plt.grid(True)
plt.xlabel("x1")
plt.ylabel("x2")

# 오른쪽: 히든 레이어 공간 (Hidden)
plt.subplot(1, 2, 2)
plt.title("2. Hidden Layer (Sigmoid Result)")
plt.scatter(H[:, 0], H[:, 1], c=색깔, s=100, alpha=0.7)
plt.grid(True)
plt.xlabel("n1")
plt.ylabel("n2")

# 시그모이드는 0~1 사이니까 박스를 그려서 범위를 표시
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='black', linestyle='--')

#구분선 긋기
plt.figure(figsize=(6, 6))
plt.title(f"Auto-Calculated Boundary (b3 = {auto_b3:.2f})")
plt.scatter(H[:, 0], H[:, 1], c=색깔, s=100, alpha=0.6, edgecolors='black')

# 자동 계산된 b3로 선 그리기
x_line = np.linspace(0, 1.2, 100)
y_line = -(w31 * x_line + auto_b3) / w32

plt.plot(x_line, y_line, 'm--', linewidth=3, label='Auto Boundary')
plt.legend()
plt.grid(True)
plt.xlabel("n1")
plt.ylabel("n2")
plt.tight_layout()
plt.show()