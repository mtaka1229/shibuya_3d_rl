import pandas as pd 
import networkx as nx
import os
import datetime as dt
import numpy as np
from datetime import timedelta 

# RSSIと検出半径DDRの関数を決める必要がある（要検討）
def ddr(rssi):
    return (-1)*rssi-56 # Bluetooth の RSSI を用いた位置推定手法の検討(仙台高専電気システム工学科) ○片上剛 本郷哲

# RSSIと尤度の関数を決める必要がある（要検討）
def likelihood(rssi):
    return rssi/50+2 
    # とりあえず-50で確率1, -100で確率0の線形関数を仮定


# reading NW data
df_node = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_node2.csv")
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_link2.csv")

df = pd.read_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/24.csv") 
result = []
# 10秒間隔でtimestepをふる
df['TIME'] = pd.to_datetime(df['TIME'])

df['time_step'] = 1 # 初期値
for i in range(1, len(df)):
    time_diff = df.loc[i, 'TIME'] - df.loc[i - 1, 'TIME']
    if time_diff >= timedelta(seconds=10):
        df.loc[i, 'time_step'] = df.loc[i - 1, 'time_step'] + 1

    """
    start_time = df.loc[0, 'TIME']
    df['time_step'] = (df['TIME'] - start_time).dt.total_seconds() // 10
    """

route = []
grouped = df.groupby('time_step')

for timestep, group in df.groupby('time_step'):
    t_group = grouped.get_group(timestep)
    n_t = len(t_group)

    for l in range(n_t):
        x_ap = t_group.loc[l, 'x']
        y_ap = t_group.loc[l, 'y']
        z_ap = t_group.loc[l, 'z']
        rssi = t_group.loc[l, 'RSSI']
            
        dict_all = {link_id: 0 for link_id in df_link['linkid']}
            
        for j in range(len(df_link)): # ここに制約が入るか，入らないか
            x_o = df_link.loc[j, 'x_o']
            y_o = df_link.loc[j, 'y_o']
            z_o = df_link.loc[j, 'z_o']

            x_d = df_link.loc[j, 'x_d']
            y_d = df_link.loc[j, 'y_d']
            z_d = df_link.loc[j, 'z_d']

            x_mid = df_link.loc[j, 'x_mid']
            y_mid = df_link.loc[j, 'y_mid']
            z_mid = df_link.loc[j, 'z_mid']

            # 距離を3種類計算
            d_mid = ((x_mid - x_ap)**2 + (y_mid - y_ap)**2 + (z_mid - z_ap)**2)**0.5
            d_o = ((x_o - x_ap)**2 + (y_o - y_ap)**2 + (z_o - z_ap)**2)**0.5
            d_d = ((x_d - x_ap)**2 + (y_d - y_ap)**2 + (z_d - z_ap)**2)**0.5

                # 3種類のうち最小のものがddrより内側にあれば良い
            d = min(d_mid, d_o, d_d)

            if d < ddr(rssi):
                dict_all[df_link.loc[j, 'linkid']] += likelihood(rssi) # 書き方合ってるはず

    linkid = max(dict_all, key=dict_all.get) # これが推定されたlinkのlink id
    print(linkid)
    route.append(linkid)
    
print(route)
