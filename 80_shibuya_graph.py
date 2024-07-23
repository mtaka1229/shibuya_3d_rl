##############################################################################################################################################
################################### BLEよりも細かい3DNWを作る．主に階段，通路レベルで．とりあえず試作版(7/10-) ########################################
##############################################################################################################################################

import matplotlib.pyplot as plt 
import numpy as np 
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd 
from PIL import Image
import datetime

# グラフを保存するファイルを日時ごとに作成（なければ作る）
graph_folder = "/Users/takahiromatsunaga/res2023/shibuya_nw"
if not os.path.exists(graph_folder):
    os.mkdir(graph_folder)

# 可視化の対象となるファイルを決めて読み込む
# file_path = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/alladdress/996.csv"
# df = pd.read_csv(file_path)

# 各mac addressに対応した出力先フォルダを作成（file_path, dfとの整合を確認）
# new_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/graph/996"
# if not os.path.exists(new_folder):
    # os.mkdir(new_folder)

###############################################################################
################ 01. 渋谷駅構内の3次元NW表現＆時系列でのRSSI強度表示 #################
###############################################################################

# ネットワークの作成
G = nx.Graph()

# ノードの追加．工事後はいくつか消えるがグラフ上はあっても問題ないので共通とする．node数は（一応）72としている．
G.add_nodes_from(i for i in range(1, 73))

# エッジの追加（一応繋げておく）最初は平面のリンクを指定し，その後で上下方向のリンクを指定する
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), 
                  (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),
                  (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), 
                  (39, 40), (40, 41), (41, 42), (42, 43), (44, 45), (45, 46), (47, 48), 
                  (38, 41), (40, 44), (41, 45), (42, 46), (44, 47), (45, 48), 
                  (49, 50), (50, 51), (51, 52), (53, 54), (54, 55), (57, 58), 
                  (49, 54), (54, 58), (51, 56), (56, 59), (59, 60),
                  (61, 62), (62, 63), (65, 66), (66, 67), (68, 69), (69, 70), 
                  (63, 65), (65, 68), (66, 69), (64, 67), (67, 70), (68, 71), (70, 72),
                  (5, 39), (6, 50), (8, 41), (10, 61), (12, 63), (17, 53), (19, 55), (21, 45), (22, 46), (26, 66), (28, 57), (29, 58), (31, 47), (32, 48), (37, 69)])

# 各ノードの座標を指定
pos = {1: (-2, -1, 0),
       2: (-1, -1, 0),
       3: (0, -1, 0),
       4: (1, -1, 0),
       5: (2, -1, 0),
       6: (3, -1, 0),
       7: (4, -1, 0),
       8: (5, -1, 0),
       9: (6, -1, 0),
       10: (7, -1, 0),
       11: (8, -1, 0),
       12: (9, -1, 0),
       13: (10, -1, 0),
       14: (-2, 0, 0),
       15: (-1, 0, 0),
       16: (0, 0, 0),
       17: (1, 0, 0),
       18: (2, 0, 0),
       19: (3, 0, 0),
       20: (4, 0, 0),
       21: (5, 0, 0),
       22: (6, 0, 0),
       23: (7, 0, 0),
       24: (8, 0, 0),
       25: (9, 0, 0),
       26: (10, 0, 0),
       27: (0, 1, 0),
       28: (1, 1, 0),  
       29: (2, 1, 0),
       30: (3, 1, 0),
       31: (4, 1, 0),
       32: (5, 1, 0),
       33: (6, 1, 0),
       34: (7, 1, 0),
       35: (8, 1, 0),
       36: (9, 1, 0),
       37: (10, 1, 0), # ここまで2F
       38: (5, -2, 1),   
       39: (3, -1, 1),
       40: (4, -1, 1),   
       41: (5, -1, 1),
       42: (6, -1, 1),   
       43: (7, -1, 1),
       44: (4, 0, 1),   
       45: (5, 0, 1),
       46: (6, 0, 1),   
       47: (4, 1, 1),
       48: (5, 1, 1), # ここまで3F
       49: (2, -1, -1),
       50: (3, -1, -1),   
       51: (4, -1, -1),
       52: (5, -1, -1),   
       53: (1, 0, -1),
       54: (2, 0, -1),   
       55: (3, 0, -1),
       56: (4, 0, -1),   
       57: (1, 1, -1),   
       58: (2, 1, -1),
       59: (4, 1, -1),
       60: (4, 2, -1), # ここまで南改札
       61: (7, -1, -1), # ここからハチ公改札
       62: (8, -1, -1),   
       63: (9, -1, -1),
       64: (11, -1, -1),
       65: (9, 0, -1),
       66: (10, 0, -1),
       67: (11, 0, -1),   
       68: (9, 1, -1),
       69: (10, 1, -1),
       70: (11, 1, -1),
       71: (9, 2, -1),
       72: (11, 2, -1)} # ここまで1F

# グラフのプロット

# プロットの作成
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
for edge in edges:
    u, v = edge
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 'k-')

# 軸の範囲の設定
ax.set_xlim([-2, 11])
ax.set_ylim([-2, 2])
ax.set_zlim([-1, 1])

# グラフを表示
plt.title("Shibuya.Sta NW")

# グラフをPNG形式で保存
file_path = os.path.join(graph_folder, "渋谷駅構内ネットワーク")
plt.savefig(file_path)

plt.show()