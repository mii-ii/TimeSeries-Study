import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    # [x1좌표, x2좌표]
    [0.1, 0.2], [0.2, 0.1], [0.1, 0.1], [0.3, 0.2], [0.2, 0.3], [0.3, 0.3], # (0,0) 근처
    [0.8, 0.9], [0.9, 0.8], [0.9, 0.9], [0.7, 0.8], [0.8, 0.7], [0.9, 0.7], # 둘 다 큰 값
    [0.1, 0.9], [0.2, 0.8], [0.1, 0.8], [0.2, 0.9], [0.3, 0.8], [0.1, 0.7], # x1은 작고, x2는 큼
    [0.9, 0.1], [0.8, 0.2], [0.8, 0.1], [0.9, 0.2], [0.8, 0.3], [0.7, 0.1]  # x1은 크고, x2는 작음
])

# 데이터 개수가 맞는지 확인 (24개여야 함)
print(f"데이터 개수: {len(X)}개")

# 색깔 구분 (두 좌표를 더해서 1.0보다 크면 빨강, 아니면 파랑)
색깔 = np.where(np.sum(X, axis=1) > 1.0, 'red', 'blue')

# 가중치 (W): 값을 비틀고 늘리는 역할
W = np.array([[ 2.0, -1.0],
              [-1.0,  2.0]])

# 바이어스 (b): 값을 이동시키는 역할
b = np.array([-0.5, -0.5])

# (1) 선형 계산: 입력 X에 가중치 W를 곱하고 바이어스 b 더하기
계산된_값 = np.dot(X, W) + b

# (2) 활성화 함수 (ReLU): 0보다 작은 마이너스 값은 전부 0으로 만들기
최종_결과 = np.maximum(0, 계산된_값)

# 그림 1: 입력 데이터 (원래 위치)
plt.figure(figsize=(5, 5))
plt.title("1. Input Data")
plt.scatter(X[:, 0], X[:, 1], c=색깔, s=100)
plt.xlim(0, 1) # 축 범위를 0~1로 고정
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# 그림 2: 히든 레이어 통과 후 (변환된 위치)
plt.figure(figsize=(5, 5))
plt.title("2. Hidden Layer (ReLU Result)")
plt.scatter(최종_결과[:, 0], 최종_결과[:, 1], c=색깔, s=100)

# 검은색 십자가 선 그리기
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid(True)
plt.show()