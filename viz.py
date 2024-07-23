import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt 

vizu_data = pd.read_csv('/Users/masudasatoki/Desktop/vizu.csv')
df_link = pd.read_csv('/Users/masudasatoki/Downloads/stanw_link_prior_all.csv')
node_data = pd.read_csv('/Users/masudasatoki/Downloads/stanw_node_all.csv')

# 保存用ディレクトリの定義
output_dir = '/Users/masudasatoki/Desktop/network_visualizations'
os.makedirs(output_dir, exist_ok=True)

# ノードデータから位置情報を抽出
node_positions = {row['nodeid'].astype(int): (row['x'], row['y']) for idx, row in node_data.iterrows()}

# 各時間帯ごとにネットワーク図を作成し保存
for time_index in range(vizu_data.shape[0]):
    # ネットワークグラフの作成
    G = nx.Graph()

    # ノードの追加
    for node in node_positions:
        G.add_node(node, pos=node_positions[node])
        G.nodes[node]['color'] = 'skyblue'
        #ノード番号が1000以上のものはノードの色を赤にする
        if 1001 <=node <= 1005:
            G.nodes[node]['color'] = 'red'
        node_color = [node["color"] for node in G.nodes.values()]

    # リンク（エッジ）とその太さを時間帯別に設定
    # 最初の時間帯（行）の交通量データを使用
    traffic_values = vizu_data.iloc[time_index]
    edge_widths = [traffic_values[str(i)] / 5 + 0.5 for i in range(1, len(traffic_values) + 1)]  # 交通量に応じた太さ
    #エッジの色を設定
    edge_colors = ['red' if edge_width > 1 else 'black' for edge_width in edge_widths]
    
    # エッジの追加
    for idx, row in df_link.iterrows():
        G.add_edge(row['o'], row['d'], weight=edge_widths[idx], color=edge_colors[idx])

    # ネットワークの描画
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=100, font_size=8,
            width=list(nx.get_edge_attributes(G, 'weight').values()))
    #plt.title(f"Network Visualization for Time Period {time_index}")
    plt.savefig(f"{output_dir}/network_time_{time_index}.png")
    plt.close()