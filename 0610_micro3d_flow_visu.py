##### 交通量の値をそのまま3Dに可視化するコード #####
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os  # osモジュールをインポート
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# ++++++++++++++++++++++++
# 時期を選ぶ(前か後か)
#period = 'post'
#period = 'prior'
#period = 'under_const'
# period = 'kintou'
#_period = 'haichi_link_7_10'
#period = 'haichi_link_10_13'

num = 2
scale = 30
color_map_is = 'magma_r'
'''
lower_bound = 0.00  # 範囲の下限
upper_bound = 1.0  # 範囲の上限
'''
lower_bound = 0.35  # 範囲の下限
upper_bound = 1.0  # 範囲の上限

#dir_path = '/Users/masuhashikana/Desktop/PythonCODEs_B4/matsunaga/'

# ++++++++++++++++++++++++

# CSVファイルからデータフレームを読み込む
# link_path = dir_path +'data/assign/' + period + '/link.csv'
# node_path = dir_path +'data/assign/' + period +'/node.csv'
# flow_path = dir_path +'data/assign/' + period +'/flow.csv'
point_gate = 2
point_amount = 40
point_sign_rate = 40
links = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_link_post_all_newminami.csv')
nodes = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_node_all_newminami.csv')
flow = pd.read_csv(f'/Users/takahiromatsunaga/res2023/results/0610multiscale/micro_flow_{point_gate}_{point_amount}point_{point_sign_rate}percent.csv')
folder_name = f'scale{scale}_gate{point_gate}_point{point_amount}_signrate{point_sign_rate}'  #フォルダ名称を指定!

# links データフレームの 'linkid' 列を整数型に変換
links['linkid'] = links['linkid'].astype(int)


# 列名称が空欄の列と`t`の列を削除
flow.drop(columns=[flow.columns[0], 't'], inplace=True)
print(flow.head())

# 混雑度の計算とカラーマップの設定
def calculate_congestion(flow, width):
    return flow / width if width != 0 else 0

# 混雑度の最大値と最小値を計算する関数
def calculate_global_max_min_congestion(links, flow):
    max_cong = 0
    min_cong = float('inf')
    for _, row in flow.iterrows():
        for _, link in links.iterrows():
            link_id = int(link['linkid'])  # リンクIDを文字列として取得
            congestion = calculate_congestion(row[link_id-1], link['width'])
            max_cong = max(max_cong, congestion)
            min_cong = min(min_cong, congestion)
    return max_cong, min_cong

# 混雑度の最大値と最小値を計算
max_congestion, min_congestion = calculate_global_max_min_congestion(links, flow)


# カラーマップの設定
cmap = cm.get_cmap(color_map_is)

# 各行（タイムステップ）ごとに可視化
# 交通量データの処理
for time_step, flow_at_time_step in flow.iterrows():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for _, link in links.iterrows():
        origin = nodes.loc[nodes['nodeid'] == link['o']].iloc[0]
        destination = nodes.loc[nodes['nodeid'] == link['d']].iloc[0]


        link_id = int(link['linkid'])  # link['linkid'] を整数型にキャスト
        print(link_id)
        # 可視化の線の太さ=交通量(flowの値)
        line_width = flow_at_time_step[link_id - 1] / scale # 交通量のスケールに応じて調整
        # 可視化の色 = 混雑度(conegstion)
        width = int(link['width'])
        congestion = calculate_congestion(flow_at_time_step[link_id - 1], width)
        print('congestion:',congestion)        
        # 混雑度に基づいて色を決定
        # カラーマップの使用範囲を0.5から0.8に限定
        normalized_congestion = (congestion - min_congestion) / (max_congestion - min_congestion)
        # カラーマップの範囲を動的に指定
        color_value = (upper_bound - lower_bound) * normalized_congestion + lower_bound
        color = cmap(color_value)
        
        ax.plot([origin['x'], destination['x']], [origin['y'], destination['y']], [origin['floor'], destination['floor']], linewidth=line_width, color=color)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Time Step: {time_step}')

    # カラーマップの凡例を追加
    # カラーマップの範囲をlower_boundとupper_boundに合わせる
    cmap_norm = Normalize(vmin=lower_bound, vmax=upper_bound)
    cmap_sm = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)
    #cmap_norm = Normalize(vmin=min_congestion, vmax=max_congestion)
    #cmap_sm = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)
    cbar = plt.colorbar(cmap_sm, ax=ax, shrink=0.7)
    cbar.ax.tick_params(labelsize='small')
    cbar.set_label('congestion', fontsize='small')

    # 太さの凡例を作成
    for line_width in [1, 3, 5]:  # 代表的な線の太さ
        ax.plot([], [], color='gray', linewidth=line_width, 
                #label=f'{line_width * scale} traffic flow')
                label=f'{line_width * scale}')
    # 凡例を表示
    ax.legend(title='line width', loc=(1.05, 0.9), fontsize='small')

    # 視点の角度を設定
    ax.view_init(azim=45, elev=30)  # 例として方位角を45度、仰角を30度に設定

    # フォルダパスを組み立て
    save_folder_path = os.path.join('/Users/takahiromatsunaga/res2023/visualization/0610micro_visu', folder_name)  # 'new_folder'は必要に応じて変更
    # フォルダが存在しない場合は作成
    os.makedirs(save_folder_path, exist_ok=True)
    file_name = f'flow_vis_timestep_{time_step}_{num}.png'
    save_path = os.path.join(save_folder_path, file_name)

    plt.savefig(save_path)  # 画像として保存
    #plt.close()
