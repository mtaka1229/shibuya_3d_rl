import pandas as pd 
import networkx as nx
import os
import datetime
import numpy as np
from datetime import timedelta 
import csv

# a function generating connection matrix from link data
def link_connection(link_data): # link_data is a dataframe
    n = len(link_data)
    A = np.eye(n) # origin of connection matrix(対角成分が1)
    for i in range(n):
        O = link_data.loc[i, 'O']
        D = link_data.loc[i, 'D']
        for j in range(n):
             # if (link_data.loc[j, 'O'] == O or D) or (link_data.loc[j, 'D'] == O or D): この書き方ダメらしい
            if ((link_data.loc[j, 'O'] == O) or (link_data.loc[j, 'O'] == D)) or (link_data.loc[j, 'D'] == O) or (link_data.loc[j, 'D'] == D):
                A[i, j] = 1
    return(A) 

# a function generating the link_id list of candidates
def link_candidate(link_data, linkid):
    A = link_connection(link_data) # 用意するリンク接続行列
    b = np.zeros(len(link_data))
    b[linkid-1] += 1
    candidate_bector = A.dot(b) # 積はdot
    true_index = np.where(candidate_bector == 1)[0] + 1 # linkidは1から始まるのでindex+1にする．リスト形式
    filtered_df = link_data[link_data['linkid'].isin(true_index)] # これ単に抜き出しているだけなので元のindexが保たれてしまう
    filtered_df = filtered_df.reset_index() # indexリセットする

    return(filtered_df)

# a function culculating DDR from RSSI
def ddr(rssi):
    return ((-1)*rssi-56)/20 # 20mを座標空間の1単位としているので，これでスケールが合う
# Bluetooth の RSSI を用いた位置推定手法の検討(仙台高専電気システム工学科) ○片上剛 本郷哲

# a function culculating likelihood from RSSI
def likelihood(rssi):
    return rssi/50+2 # とりあえず-50で確率1, -100で確率0の線形関数を仮定

# reading NW data
df_node = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_node2.csv")
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_link2.csv")

df = pd.read_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/1.csv")
df['TIME'] = pd.to_datetime(df['TIME']) # datetime形式に変換

df['time_step'] = 1 # initial value for all the row

# time_stepを振り分ける
start_time = df.loc[0, 'TIME'] # 一番最初のデータを開始時刻の初期値にしておく．forループでstart_timeを入れ替える
for i in range(1, len(df)):
    time_diff = (df.loc[i, 'TIME'] - start_time).total_seconds() # total_seconds()で秒変換され，int型のデータになっているはず
    if time_diff >= 10:
        df['time_step'].iloc[i:] += 1 # i番目以降の全部に+1した方がやりやすい
        start_time = df.loc[i, 'TIME'] # start_timeは更新

#print(df)

# list of the determined link_id
result = []

# initial candidates sets
A = df_link # Aは最初の候補なのでdf_linkそのもの
C =[A] # initial element is A, which inclueds all the links data.Cはリンクデータ

# grouping depending on 'timestep' values
grouped = df.groupby('time_step')
# grouped = grouped.reset_index() # groupbyメソッドで分割してもindexは通しのままになる

# making the list of the dataframes
dataframes_list = [group.reset_index(drop=True) for name, group in grouped]

for i in range(len(dataframes_list)):
    #print(f'今からtimestep_{i}の作業をやります')
    df_at_time = dataframes_list[i] 
    cand_df_link = C[i] 

    #print(df_at_time)
    #print(cand_df_link)

    # csnd_df_link['linkid']をkeyとして追加し，初期値を0にする
    dict_for_each_timestep = {link_id: 0 for link_id in cand_df_link['linkid']} # この書き方できるように！！

    for j in range(len(df_at_time)):
        # RSSIを取得
        rssi = df_at_time.loc[j, 'RSSI']

        # 各データiの検出bleの座標を取得
        x_ap = df_at_time.loc[j, 'x']
        y_ap = df_at_time.loc[j, 'y']
        z_ap = df_at_time.loc[j, 'z']

        for k in range(len(cand_df_link)): 
            #print(f'今から{i}番目{k}番目のリンクを見ます')
            x_o = cand_df_link.loc[k, 'x_o'] 
            y_o = cand_df_link.loc[k, 'y_o']
            z_o = cand_df_link.loc[k, 'z_o']

            x_d = cand_df_link.loc[k, 'x_d']
            y_d = cand_df_link.loc[k, 'y_d']
            z_d = cand_df_link.loc[k, 'z_d']

            x_mid = cand_df_link.loc[k, 'x_mid']
            y_mid = cand_df_link.loc[k, 'y_mid']
            z_mid = cand_df_link.loc[k, 'z_mid']

            # 距離を3種類計算
            d_mid = ((x_mid - x_ap)**2 + (y_mid - y_ap)**2 + (z_mid - z_ap)**2)**0.5
            d_o = ((x_o - x_ap)**2 + (y_o - y_ap)**2 + (z_o - z_ap)**2)**0.5
            d_d = ((x_d - x_ap)**2 + (y_d - y_ap)**2 + (z_d - z_ap)**2)**0.5

            # 3種類のうち最小のものがddrより内側にあれば良い
            d = min(d_mid, d_o, d_d)

            if d < ddr(rssi):
                dict_for_each_timestep[cand_df_link.loc[k, 'linkid']] += likelihood(rssi)

        #print(f'{j}番目のデータは読み込めています')

    # 最もvalueの大きいkeyを返す．それが初期linkになる
    estimeted_linkid = max(dict_for_each_timestep, key=dict_for_each_timestep.get) 
    result.append(estimeted_linkid) 
    next_chice_set = link_candidate(df_link, estimeted_linkid)
    C.append(next_chice_set)

    #print(estimeted_linkid) # 候補集合の中に該当するリンクが見当たらないのか？
    #print(next_chice_set)
    #print(C)

    #print(f'{i}番目のtime_stepの塊は処理できています')

# print(result)
with open(f'/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/1_result.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(result)