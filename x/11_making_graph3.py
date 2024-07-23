# 図面ベースで作ったNWの可視化
# pythonの図がわかりにくければCADとかを勉強して建築的な要素を表現した方がうぉーーーーーーーってなるのは，それはそう．
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd 
from PIL import Image
import datetime

# まずノードのデータを作る
# 各ノードの座標
# この修正面倒くっさ
pos = {1: (1, -1, 0), # 2F
       2: (2.5, -1, 0),
       3: (6, -1, 0),
       4: (9, -1, 0),
       5: (0, 0, 0),
       6: (2, 0, 0),
       7: (5.5, 0, 0),
       8: (8, 0, 0),
       9: (10, 0, 0),
       10: (0, 1, 0),
       11: (1.5, 1, 0),
       12: (4.5, 1, 0),
       13: (8, 1, 0),
       14: (10, 1, 0),
       15: (5, 1, 1), # 3F
       16: (5, 0, 1),
       17: (5, -1, 1),
       18: (6, -1, 1),
       19: (1.5, 1, -1), # 1F南
       20: (1.5, 0, -1),
       21: (1.5, -1, -1),
       22: (4, -1, -1),
       23: (4, 1, -1),
       24: (5, -1, -1),
       25: (8, -1, -1), # 1Fハチ公
       26: (9, -1, -1),
       27: (10, 0.5, -1),
       28: (9, 1, -1),  
       29: (11.5, 1, -1),
       30: (11, -1, -1),} # ここまで

nodeid = pos.keys()

# print(nodeid)

# 注意：dict.value()で取得できるのはobjectなのでlist()によってlist化しないと使えない
values = list(pos.values())

# print(values)

X = []
Y = []
Z = []

for i in range(len(values)):
    X.append(values[i][0])
    Y.append(values[i][1])
    Z.append(values[i][2])

# データを辞書型で定義
data = {
    "nodeid": [i for i in range(1, 31)],
    "x": [x for x in X],
    "y": [y for y in Y],
    "z": [z for z in Z]
}

# DataFrameを作成
df = pd.DataFrame(data)

# DataFrameの中身を表示
# print(df)

# 名前はnode2とする
df.to_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_node2.csv")

# 次にリンクのデータを作る

# リンクの追加（一応繋げておく）最初は平面のリンクを指定し，その後で上下方向のリンクを指定する
linklist = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (13, 14), #2F
                (15, 16), (16, 17), (17, 18), #3F
                (19, 20), (20, 21), (21, 22), (22, 23), (22, 24), #1F南
                (25, 26), (26, 27), (27, 28), (27, 29), (27, 30), #ここまで平面のリンク
                (2, 17), (3, 17), (7, 16), (12, 15), #2F→3Fのリンク
                (2, 21), (6, 20), (11, 19), 
                (14, 27), (9, 27), (4, 26), (3, 25)]

O = []
D = []

for i in range(len(linklist)):
    O.append(linklist[i][0])
    D.append(linklist[i][1])

# データを辞書型で定義
data2 = {
    "linkid": [i for i in range(1, 36)],
    "O": [o for o in O],
    "D": [d for d in D],
    "width": [1]*35,
    "length":[1]*35
}

# DataFrameを作成
df2 = pd.DataFrame(data2)

# DataFrameの中身を表示
# print(df2)

df2.to_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_link2.csv")

### 可視化で確認
# 保存先
graph_folder = "/Users/takahiromatsunaga/res2023/shibuya_nw"

G = nx.Graph()

# ノードの追加．工事後はいくつか消えるがグラフ上はあっても問題ないので共通とする．node数は（一応）72としている．
G.add_nodes_from(i for i in range(1, 31))
G.add_edges_from(linklist)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ノードのプロット
nodes = list(G.nodes())
x = [pos[node][0] for node in nodes]
y = [pos[node][1] for node in nodes]
z = [pos[node][2] for node in nodes]

# ノードの色を反映してプロット
cax = ax.scatter(x, y, z, cmap='coolwarm', s=200) #s=200ってどういう意味
#fig.colorbar(cax, ax=ax, label='RSSI')

# エッジのプロット
edges = list(G.edges())
edges_label = data2["linkid"].values()

for edge in edges:
    u, v = edge
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 'k-')

# 軸の範囲の設定
ax.set_xlim([-2, 11])
ax.set_ylim([-2, 2])
ax.set_zlim([-1, 1])

# グラフを表示
plt.title("Shibuya.Sta NW2")

# グラフをPNG形式で保存
file_path = os.path.join(graph_folder, "渋谷駅構内ネットワーク2")
plt.savefig(file_path)

plt.show()
"""
# 可視化でチェック
# 3Dプロットの作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# 軸ラベルの設定
ax.set_xlabel('X軸')
ax.set_ylabel('Y軸')
ax.set_zlabel('Z軸')

# グラフを表示
plt.show()
"""
