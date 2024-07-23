import pandas as pd 

# df = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_link.csv')

# max = df['length'].max()
# min = df['length'].min()
# mean = df['length'].mean()
# sigma = (df['length'].var())**0.5

# print(max, min, mean, sigma)

"""
import pandas as pd
import matplotlib.pyplot as plt

# # 'length'列の値の頻度分布図を作成
# plt.figure(figsize=(10, 6))
# plt.hist(df['length'], bins=30, edgecolor='black')

# # グラフのタイトルとラベルを設定
# plt.title('Frequency Distribution of Length Values')
# plt.xlabel('Length')
# plt.ylabel('Frequency')

# # グラフを表示
# plt.show()

# 'length'列の値の頻度分布図を作成
plt.figure(figsize=(10, 6))

# ヒストグラムを累積モードで作成
plt.hist(df['length'], bins=30, edgecolor='black', cumulative=True, density=True)

# グラフのタイトルとラベルを設定
plt.title('Cumulative Distribution of Length Values')
plt.xlabel('Length')
plt.ylabel('Cumulative Frequency')

# グラフを表示
plt.show()
"""

"""
import matplotlib.pyplot as plt
# 各user idの登場回数をカウント
df = pd.read_csv('/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_webq/q_ans/diary.csv', encoding='shift-jis')
user_counts = df['id'].value_counts()

# 可視化
plt.figure(figsize=(10, 6))
user_counts.plot(kind='bar')
plt.xlabel('User ID')
plt.ylabel('Count')
plt.title('User ID Count Distribution')
plt.show()
nuser = df['id'].nunique()
print('nuser', nuser)
"""

# 渋谷駅近辺のlink-based なnetwork dataを作成
"""
import pandas as pd 
import csv

df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_link.csv')
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_node.csv')

# network data: どのリンクとどのリンクが繋がっているか
df_network = pd.DataFrame(columns=['k', 'a'])
for i in range(len(df_link)):
    connected_links = [] # このリンクに接続してるリンク
    linkid = df_link.loc[i, 'id']

    onode = df_link.loc[i, 'onode']
    dnode = df_link.loc[i, 'dnode']

    temp_links = []

    # 条件に基づくフィルタリングとIDの追加
    if not df_link[df_link['onode'] == onode].empty:
        temp_links.extend(df_link[df_link['onode'] == onode]['id'].tolist())

    if not df_link[df_link['dnode'] == onode].empty:
        temp_links.extend(df_link[df_link['dnode'] == onode]['id'].tolist())

    if not df_link[df_link['onode'] == dnode].empty:
        temp_links.extend(df_link[df_link['onode'] == dnode]['id'].tolist())

    if not df_link[df_link['dnode'] == dnode].empty:
        temp_links.extend(df_link[df_link['dnode'] == dnode]['id'].tolist())

    # 重複を避けるために一意な要素のみを残す
    temp_links = list(set(temp_links))

    # Remove the linkid if it's in the temp_links list
    temp_links = [link for link in temp_links if link != linkid] # linkidと同じやつは意味がないのでとる（o→oになって意味不明）
    connected_links.extend(temp_links)

    if linkid == 1:
        print(temp_links)
   #print(connected_links)

    for connected_link in connected_links:
        df_network = df_network.append({'k': linkid, 'a': connected_link}, ignore_index=True)

df_network.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_network.csv')

print(df_network) # network接続データ作れた，，（matrix data）
"""


# node-basedなnetworkデータを作る

import pandas as pd 
import csv

df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_link.csv')
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_node.csv')

# network data: どのノードとどのノードが繋がっているか
df_network = pd.DataFrame(columns=['k', 'a'])
for i in range(len(df_node)):
    connected_nodes = [] # このnodeに接続してるnode
    nodeid = df_node.loc[i, 'id'] # このノードのnodeid

    temp_nodes = []

    # 条件に基づくフィルタリングとIDの追加
    if not df_link[df_link['onode'] == nodeid].empty:
        temp_nodes.extend(df_link[df_link['onode'] == nodeid]['dnode'].tolist())

    if not df_link[df_link['dnode'] == nodeid].empty:
        temp_nodes.extend(df_link[df_link['dnode'] == nodeid]['onode'].tolist())

    # if not df_link[df_link['onode'] == dnode].empty:
    #     temp_links.extend(df_link[df_link['onode'] == dnode]['id'].tolist())

    # if not df_link[df_link['dnode'] == dnode].empty:
    #     temp_links.extend(df_link[df_link['dnode'] == dnode]['id'].tolist())

    temp_nodes =  list(set(temp_nodes)) # 隣接のーどがtemp_nodesに入ってるので

    # Remove the linkid if it's in the temp_links list
    temp_nodes = [node for node in temp_nodes if node != nodeid] # linkidと同じやつは意味がないのでとる（o→oになって意味不明）
    connected_nodes.extend(temp_nodes)

    for connected_node in connected_nodes:
        df_network = df_network.append({'k': nodeid, 'a': connected_node}, ignore_index=True)

df_network.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/map/shibuya_mezzo_nodebased_network.csv')


# import os 
# folder = '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/shibuya_mezzo_gps4'
# list = os.listdir(folder)

# print(len(list))