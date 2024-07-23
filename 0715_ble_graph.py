import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
print('aaa')
# サンプルデータの作成（ビーコンの位置と観測値）
beacon_positions = np.array([
    [0, 0], [1, 0], [0, 1], [1, 1]
])
observations = {
    "mac1": np.array([
        [0.9, 0.7, 0.8, 0.6],
        [0.8, 0.9, 0.7, 0.8],
        [0.7, 0.6, 0.9, 0.7]
    ])
}
print(observations)
TT = 3  # 時刻の数

B = 4   # ビーコンの数
print(f'TT={TT}')
# 特徴量抽出（観測値をそのまま利用）
features = observations["mac1"]

# 次元削減（PCA）
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

print('reduced_features', reduced_features) # 2dに減らしたってど言う意味だろう

# LSTMモデルの定義
model = Sequential()
model.add(LSTM(50, input_shape=(TT-1, reduced_features.shape[1]), return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(2))  # 移動ベクトルの次元（x, y）
print('ここまで')
model.compile(optimizer='adam', loss='mse')

# 訓練データの準備
X_train = np.array([reduced_features[:-1]])
y_train = np.array([reduced_features[1:] - reduced_features[:-1]])

# モデルの訓練
model.fit(X_train, y_train, epochs=100, verbose=1)

# 移動ベクトルの推定
predicted_movement = model.predict(X_train)

# 結果の表示
print("Predicted movement vectors:")
print(predicted_movement)


######### 距離構造を明記

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# サンプルデータの作成（ビーコンの位置と観測値）
beacon_positions = np.array([
    [0, 0], [1, 0], [0, 1], [1, 1]
])
observations = {
    "mac1": np.array([
        [0.9, 0.7, 0.8, 0.6],
        [0.8, 0.9, 0.7, 0.8],
        [0.7, 0.6, 0.9, 0.7]
    ])
}

TT = 3  # 時刻の数
B = 4   # ビーコンの数

# 観測値から距離の推定
def estimate_distance(observations):
    distances = 1 / (observations + 1e-5)  # 観測値の逆数を取る（0での除算を避けるために小さな値を足す）
    return distances

# 重心法による位置の推定
def compute_positions(beacon_positions, distances):
    positions = []
    for t in range(distances.shape[0]):
        weights = distances[t]
        weights_sum = np.sum(weights)
        position = np.sum(weights[:, np.newaxis] * beacon_positions, axis=0) / weights_sum
        positions.append(position)
    return np.array(positions)

# 距離推定
distances = estimate_distance(observations["mac1"])

# 位置推定
positions = compute_positions(beacon_positions, distances)

# 次元削減（ここでは不要ですが例として残しています）
pca = PCA(n_components=2)
reduced_positions = pca.fit_transform(positions)

# LSTMモデルの定義
model = Sequential()
model.add(LSTM(50, input_shape=(reduced_positions.shape[0] - 1, reduced_positions.shape[1]), return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(2))  # 移動ベクトルの次元（x, y）

model.compile(optimizer='adam', loss='mse')

# 訓練データの準備
X_train = np.array([reduced_positions[:-1]])
y_train = np.array([reduced_positions[1:] - reduced_positions[:-1]])

# モデルの訓練
model.fit(X_train, y_train, epochs=100, verbose=1)

# 移動ベクトルの推定
predicted_movement = model.predict(X_train)

# 結果の表示
print("Predicted movement vectors:")
print(predicted_movement)
