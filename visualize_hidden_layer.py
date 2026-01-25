import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 생성 (총 24개, 2개의 클래스로 구분)
# 입력 데이터 x1, x2 (예: 0.25, 0.8 등)
np.random.seed(42)
N = 24
# 0~1 사이의 랜덤한 좌표값 24개 생성
X = np.random.rand(N, 2) 

# 시각화를 위해 임의로 클래스(정답)를 나눕니다. 
# (대각선을 기준으로 위쪽은 빨강, 아래쪽은 파랑)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# 2. 모델 파라미터 설정 (그림 속 W1, b1)
# 학습된 가중치라고 가정하고, 데이터를 잘 퍼뜨려줄 값을 설정합니다.
# 입력이 2개, 히든뉴런이 2개이므로 가중치는 2x2 행렬입니다.
W1 = np.array([[ 2.0, -1.5],
               [-1.5,  2.0]])

# 바이어스 (뉴런 각각에 더해지는 값)
b1 = np.array([-0.2, -0.2])

# 3. 히든 레이어 계산 과정 (상세 설명)
# (1) 선형 변환: Z = X * W + b
# 행렬 곱셈을 통해 데이터를 회전/이동시킵니다.
Z = np.dot(X, W1) + b1

# (2) 활성화 함수 (ReLU) 통과
# 교수님이 말씀하신 "한쪽으로 몰리는 현상"을 보기 위해 ReLU를 씁니다.
# 0보다 작은 값은 모두 0이 되어 축에 달라붙게 됩니다.
H = np.maximum(0, Z)  # Hidden Layer Output (n1, n2)

# 4. 시각화 (좌표평면 그리기)
plt.figure(figsize=(12, 5))

# [왼쪽 그림] 입력 공간 (Input Space)
plt.subplot(1, 2, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Class 0', s=100, alpha=0.7)
plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1', s=100, alpha=0.7)
plt.title("1. Input Layer (x1, x2)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.legend()

# [오른쪽 그림] 히든 레이어 공간 (Hidden Space)
plt.subplot(1, 2, 2)
plt.scatter(H[y==0, 0], H[y==0, 1], color='blue', label='Class 0', s=100, alpha=0.7)
plt.scatter(H[y==1, 0], H[y==1, 1], color='red', label='Class 1', s=100, alpha=0.7)
plt.title("2. Hidden Layer (n1, n2)")
plt.xlabel("n1 (Neuron 1)")
plt.ylabel("n2 (Neuron 2)")
plt.grid(True)
# 0,0 근처에 몰린 것을 잘 보기 위해 축 설정
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.legend()

plt.tight_layout()
plt.show()
