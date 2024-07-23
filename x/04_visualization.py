##############################################################################################################################################
################################### 03_cleaningでクリーニングしたalladdress内のmac addressを使って可視化する ########################################
###################### ただし全てのmac addressを可視化してもしょうがないのでいくつか抜き出して可視化する．故に処理はファイルごととする ########################
########### 一つのmac addressを複数の切り口から可視化するのが大切なので，同時に複数の画像を作り，mac addressごとに画像フォルダを使ってファイリングする ###########
##############################################################################################################################################

import matplotlib.pyplot as plt 
import numpy as np 
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd 
from PIL import Image
import datetime
# /Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/4.csv
# グラフを保存するファイルを日時ごとに作成（なければ作る）
graph_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/graph"
if not os.path.exists(graph_folder):
    os.mkdir(graph_folder)

# 可視化の対象となるファイルを決めて読み込む
file_path = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/20.csv"
df = pd.read_csv(file_path)

# 各mac addressに対応した出力先フォルダを作成（file_path, dfとの整合を確認）
new_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/graph/20"
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

###############################################################################
################ 01. 渋谷駅構内の3次元NW表現＆時系列でのRSSI強度表示 #################
###############################################################################

# ネットワークの作成．ここでのNWはBLEの観測網のこと
G = nx.Graph()

# ノードの追加．工事後はいくつか消えるがグラフ上はあっても問題ないので共通とする
G.add_nodes_from(i for i in range(1, 24))

# エッジの追加(怪しいが一応)
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (8, 9), (9, 10), (10, 11), (12, 13), (22, 23), (12, 14), (14, 15), (15, 16), (4, 14), (15, 23), (17, 18), (19, 20), (20, 21), (19, 7), (21, 7)])

# 各ノードの座標を指定
pos = {1: (1, 0, 0),
       2: (2, 0, 0),
       3: (3, 0, 0),
       4: (4, 0, 0),
       5: (5, 0, 0),
       6: (6, 0, 0),
       7: (5, 0, -1),  #ここだけ1F
       8: (2, 1, 0),
       9: (3, 1, 0),
       10: (4, 1, 0),
       11: (5, 1, 0),
       12: (4, 1, 1),   #2F
       13: (6, 1, 1),
       14: (4, 0, 1),
       15: (4, -1, 1),
       16: (6, -1, 1),
       17: (1, 1, -1),   #1F
       18: (1, 0, -1),
       19: (4, 0, -1),
       20: (4, 1, -1),
       21: (5, 1, -1),
       22: (3, -1, 0),   #2F
       23: (4, -1, 0)}

#image_listの作成
image_list = []

# 時刻ごとにグラフを出力
for i in range(len(df)):
    time = df.loc[i, 'TIME']
    
    # プロットの作成
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ノードのプロット
    nodes = list(G.nodes())
    x = [pos[node][0] for node in nodes]
    y = [pos[node][1] for node in nodes]
    z = [pos[node][2] for node in nodes]

    # ノードの色を設定
    # TIME列に記載されている時刻において，id_num列の値と一致するノードに色をつけ，それ以外のノードは白色にする．この時，「RSSI」列の値に応じてグラデーションをつける

    color_map = []
    for node in nodes:
        if node == df.loc[i, 'id_num']: 
            rssi = df.loc[i, 'RSSI']
            color_map.append(rssi)
        else:
            color_map.append(-120)  # 該当しないノードは白色に設定

    # ノードの色を反映してプロット
    cax = ax.scatter(x, y, z, c=color_map, cmap='coolwarm', s=200, vmin=-120, vmax=-60)
    fig.colorbar(cax, ax=ax, label='RSSI')

    # エッジのプロット
    edges = list(G.edges())
    for edge in edges:
        u, v = edge
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], 'k-')

    # 軸の範囲の設定
    ax.set_xlim([0, 6])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-1, 1])

    # グラフを表示
    plt.title(f'Time: {time}')
    
    # 一枚ごとに保存しておくためのフォルダを作成
    image_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/graph/20/3Dnw_image"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    # グラフをPNG形式で保存
    file_path = os.path.join(image_folder, f'shibuyaNW_{time}.png')
    plt.savefig(file_path)

    # 画像をリストに追加
    image_list.append(Image.open(file_path))
    
    # 一時ファイルの削除
    os.remove(file_path)

    plt.close(fig)

# GIF動画の保存先ファイルパス
gif_filename = os.path.join(new_folder, '3Dnw_animation.gif')

# 画像リストをGIF動画として保存
image_list[0].save(gif_filename, format='GIF', append_images=image_list[1:], save_all=True, duration=200, loop=0)

# 完成したGIF動画のパスを表示
# print('GIF動画が作成されました:', gif_filename)

"""
###############################################################################
################ 02. RSSIと観測BLEidとを対応させた散布図 #################
###############################################################################

#データは冒頭で読み込み済みなのでdfをそのまま使う

# 03_cleaning.pyによりid_num, TIME列の存在は担保されている
# id_numとRSSIの関係を可視化．in_numをTIMEにすると，時間とRSSIのグラフが描ける
X = df['id_num']
Y = df['RSSI']

plt.xlabel("id_num")
plt.ylabel("RSSI")
plt.xticks(range(2, 25, 2))

plt.scatter(X, Y)

#plt.show()

rssi_scatter_name = "rssi_scatter.png"
rssi_scatter_path = os.path.join(new_folder, rssi_scatter_name)

# 保存
plt.savefig(rssi_scatter_path, format="png", dpi=300)
plt.close()

###############################################################################
################ 03. BLEidと検出回数との関係のグラフ #################
###############################################################################

# ここでもデータはすでに読み込んであるものを使うのでdfを継続して使用できる
# データフレームの設定
columns = ["id001", "id002", "id003", "id004", "id005", "id006", "id008", "id009", "id010", "id011", "id022", "id023", "id012", "id013", "id014", "id015", "id016", "id017", "id018", "id007", "id019", "id020", "id021"]

#countしたデータを入れるための空のデータフレームcsv
df_new = pd.DataFrame(columns=columns)

#平均rssiを出すための空のデータフレームcsv
df_rssi = pd.DataFrame(columns=columns)

#ここでスタートタイムを切り捨て，エンドタイムを切り上げたい，その上で1分おきに集計したい
start_time = df.loc[0, 'TIME'].replace(second=0, microsecond=0)
end_time = df.loc[df.index[-1], "TIME"].replace(second=0, microsecond=0)
end_time = end_time + datetime.timedelta(minutes=1)

# 1分ごとのタイムステップ
delta = datetime.timedelta(minutes=1)

# 最初の時間だけはdf_newに行を追加しておく
df_new.loc[start_time] = 0
df_rssi.loc[start_time] = 0

current_time = start_time

# while current_time <= end_time:←while文のせいでめちゃ無限ループエラーで大変だったので使い方には気をつける
for i in range(len(df)):
    index_time = df.loc[i, "TIME"]
    id_value = df.loc[i, 'ID']

    # TIME列の昇順に並べ替えてる前提があるので上から抑えるだけでOK
    if index_time < current_time + delta:
                
        if current_time in df_new.index:
            df_new.loc[current_time, id_value] += 1

        else:
            df_new.loc[current_time] = 0
            df_new.loc[current_time, id_value] = 1

    else:
        current_time += delta  # 現在の時間を更新
        df_new.loc[current_time] = 0
        df_new.loc[current_time, id_value] = 1

# macアドレスごとに作ったdf_newをnew_folderの直下に保存する．
df_new_name = "count.csv"
df_new_path = os.path.join(new_folder, df_new_name)

# csvに出力
df_new.to_csv(df_new_path)

# 出力したdf_newを可視化 
df_new_t = df_new.T
X2 = [i+1 for i in range(len(df_new_t.columns))]

#（棒グラフが生える）横軸の本数を指定．これはidの数
fig, axes = plt.subplots(len(df_new_t)-1) #-1しないとラベルの列の分だけ列数が追加されてしまう

for ax in axes[:-1]:
    ax.tick_params(bottom=False, labelbottom=False)  # 一番下のax以外，x軸目盛を消す
for i in range(len(axes)):
    ax = axes[i]
    row_data = df_new_t.iloc[i+1]
    Y2 = row_data.tolist()
    ax.bar(X2, Y2)  # 棒グラフ
    ax.set_yticks([10]) # y軸の目盛りを指定
    ax.set_ylim(0, 20)  # y軸の上限・下限を指定

plt.subplots_adjust(wspace=0, hspace=0)  # 間を詰める
plt.xlabel('time step')
plt.ylabel('count at each BLE')

#描画したグラフをpng形式で保存
image_name = "count.png"
image_path = os.path.join(new_folder, image_name)
plt.savefig(image_path, format="png", dpi=300)

plt.close()
"""