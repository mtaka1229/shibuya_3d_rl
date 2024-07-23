## 個人のBLE生データを駅構内NW上に可視化するコード
## base_pathやその他pathは適当に書き換える

import pandas as pd 
import os
import matplotlib.pyplot as plt 
import networkx as nx
from matplotlib.ticker import MultipleLocator
import matplotlib.cm as cm
from PIL import Image

base_path = '/Users/takahiromatsunaga/res2023'
day_stamp = '20230130'
time_stamp = '08'
daytime_stamp = f'{day_stamp}_{time_stamp}'
userid = 64 # folder内の個人を指定（適当）
df_link = pd.read_csv(os.path.join(base_path, 'shibuya_nw/shibuya_stanw/stanw_link_post.csv'))  
df_link = df_link.iloc[:-3, :] # 仮想リンク・仮想ノードは表示しない
df_node = pd.read_csv(os.path.join(base_path, 'shibuya_nw/shibuya_stanw/stanw_node.csv')) 
df_node = df_node.iloc[:-3, :]
df_ble = pd.read_csv(os.path.join(base_path, 'shibuya_nw/shibuya_nw_detail/ble_nw.csv')) 
## 工事中・後はビーコンが減っている
filtered_ble = df_ble[~df_ble['ID'].between(8, 11) & ~df_ble['ID'].between(12, 14)]
# 個人の生データ
df_log = pd.read_csv(os.path.join(base_path, f'bledata/ble_timegroup/{daytime_stamp}/{daytime_stamp}45_18sec_end162021_under10/{userid}.csv'))

# 各mac addressに対応した出力先フォルダを作成
new_folder = os.path.join(base_path, f'bledata/ble_timegroup/{daytime_stamp}/visualization') 
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

image_list = []
for l in range(len(df_log)):

    # グラフの作成
    G = nx.Graph()

    # 階ごとの配色の設定
    color_mapping = {
        14.8: 'orange',
        20.5: 'navy',
        28.5: 'orange' #'yellowgreen'
    }

    cmap = cm.get_cmap('coolwarm')  # 適切なカラーマップを選択

    # ノードの追加
    for idx, row in df_node.iterrows():
        G.add_node(row['nodeid'], pos=(row['x'], row['y'], row['floor']))

    # リンクの追加
    for idx, row in df_link.iterrows():
        floor_source = G.nodes[row['o']]['pos'][2]  # リンクの始点の'floor'値
        floor_target = G.nodes[row['d']]['pos'][2]  # リンクの終点の'floor'値
        G.add_edge(row['o'], row['d'], linkid=row['linkid']) #, color = link_color) #, linewidth = link_width, linestyle = link_style)

    isolated_nodes = [node for node in G.nodes() if G.degree[node] == 0]

    # グラフからリンクを持たないノードを削除
    G.remove_nodes_from(isolated_nodes)

    # 座標の取得
    pos = nx.get_node_attributes(G, 'pos')

    # グラフの描画
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # グリッドの設定
    ax.grid(True)

    # 各軸のメモリの間隔を10に設定
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(40))
    ax.zaxis.set_major_locator(MultipleLocator(10))

    point_size = 150

    #for l in range(len(df_log)):
    beacon_id = df_log.loc[l, 'ID'] 
    beacon_x = df_ble[df_ble['ID'] == beacon_id]['x'].iloc[0]
    beacon_y = df_ble[df_ble['ID'] == beacon_id]['y'].iloc[0]
    beacon_z = df_ble[df_ble['ID'] == beacon_id]['floor'].iloc[0]

    timestep = df_log.loc[l, 'timestep']

    rssi = df_log.loc[l, 'RSSI']
    color = cmap((rssi - (-120)) / ((-40) - (-100)))  # カラーマップから色を取得

    point_size = 1000
    ax.scatter(beacon_x, beacon_y, beacon_z, c=color, s = point_size, label='Point Cloud Data from BLE')

    ax.view_init(elev=20, azim=200)  # elevは上下方向の角度、azimは左右方向の角度
    ax.set_box_aspect([1, 1, 0.5])  # [横方向, 前後方向, 上下方向]の比率を設定

    for node, coords in pos.items():
        ax.scatter(*coords, color='skyblue')
        #ax.text(*coords, s=f"Node {node}", color='black')

    for edge in G.edges(data=True):
        node1, node2, edge_attr = edge
        x = [pos[node1][0], pos[node2][0]]
        y = [pos[node1][1], pos[node2][1]]
        z = [pos[node1][2], pos[node2][2]]

        floor_source = G.nodes[node1]['pos'][2]  # リンクの始点の'floor'値
        floor_target = G.nodes[node2]['pos'][2]  # リンクの終点の'floor'値

        # 'floor'値に基づいて色を割り当てる
        if floor_source in color_mapping and floor_target in color_mapping:
            link_color = color_mapping[floor_source]  # 始点と終点の両方がマッピング内にある場合        
            link_width = 5.0   
            link_style = '-'   
        else: 
            link_width = 5.0
            link_color = 'lightgray'
            link_style = '--'   

        ax.plot(x, y, z, color=link_color, linewidth = link_width, linestyle = link_style)

    plt.title(f'Time: {timestep}', fontsize = 50)
  
    # 一枚ごとに保存しておくためのフォルダを作成
    image_folder = os.path.join(new_folder, 'temporary')
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    # グラフをPNG形式で保存
    file_path = os.path.join(image_folder, f'ble_log{userid}_{l}.png')
    plt.savefig(file_path)

    # 画像をリストに追加
    image_list.append(Image.open(file_path))
    
    # 一時ファイルの削除
    #os.remove(file_path)

    plt.close(fig)

# GIF動画の保存先ファイルパス
gif_filename = os.path.join(new_folder, f'BLElog_animation{userid}.gif')

# 画像リストをGIF動画として保存
image_list[0].save(gif_filename, format='GIF', append_images=image_list[1:], save_all=True, duration=200, loop=0)
