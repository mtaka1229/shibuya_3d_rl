import pandas as pd 
import networkx as nx
import os
import datetime
import numpy as np
from datetime import timedelta 
import csv

# a function generating link connection matrix from link data
def link_connection(link_data):
    n = len(link_data)
    A = np.eye(n)
    for i in range(n):
        O = link_data.loc[i, 'O']
        D = link_data.loc[i, 'D']
        for j in range(n):
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
    filtered_df = link_data[link_data['linkid'].isin(true_index)]
    filtered_df = filtered_df.reset_index() # indexリセットする

    return(filtered_df)

# a function culculating DDR from RSSI
def ddr(rssi):
    return ((-1)*rssi-56)/20 # 20mを座標空間の1単位としているので，これでスケールが合う
# Bluetooth の RSSI を用いた位置推定手法の検討(仙台高専電気システム工学科) ○片上剛 本郷哲

# a function culculating likelihood from RSSI
# これは結局DDR内のリンク数Lに対して一律で1/Lの尤度を付与

# reading NW data
df_node = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_node2.csv")
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_link2.csv")

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_18/alladdress"
file_list = list(os.listdir(search_folder))

output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_18/estimated_route"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

result = []

# route estimation
for file_name in file_list:
    # 各userのパス（経路全体）の確定
    file_path = os.path.join(search_folder, file_name)

    # dataの読み込み
    df = pd.read_csv(file_path) 
    df['TIME'] = pd.to_datetime(df['TIME']) # datetime形式に変換

    # time_stepを幅10"で振り分ける
    start_time = df.loc[0, 'TIME'] # 一番最初のデータを開始時刻の初期値にしておく．forループでstart_timeを入れ替える
    df['time_step'] = 1
    for i in range(1, len(df)):
        time_diff = (df.loc[i, 'TIME'] - start_time).total_seconds() # total_seconds()で秒変換され，int型のデータになる
        if time_diff >= 10:
            df['time_step'].iloc[i:] += 1 # i番目以降の全部に+1した方がやりやすい
            start_time = df.loc[i, 'TIME'] # start_timeは更新

    # list of the determined link_id
    each_result = []

    # initial candidates sets
    A = df_link # Aは最初の候補なのでdf_linkそのもの
    C =[A] # Cはリンクデータのリスト．毎ループで追加し，次のループでそれを参照する

    # grouping depending on 'timestep' values
    grouped = df.groupby('time_step') # groupbyメソッドで分割してもindexは保たれる

    # making the list of the dataframes
    dataframes_list = [group.reset_index(drop=True) for name, group in grouped] # ここでindexをリセットする

    for i in range(len(dataframes_list)):
        # 各timestepのパス（経路全体）の確定

        df_at_time = dataframes_list[i] 
        # df_at_timeの長さはtimestepの長さ
        cand_df_link = C[i] 

        # csnd_df_link['linkid']をkeyとして追加し，初期値を0にする
        dict_for_each_timestep = {link_id: 0 for link_id in cand_df_link['linkid']} # この書き方できるように！！

        for j in range(len(df_at_time)):
            # timestep内の個々の時点jにおける最も近いリンクを捜索
            # RSSIを取得
            rssi = df_at_time.loc[j, 'RSSI']

            # 各データiの検出bleの座標を取得
            x_ap = df_at_time.loc[j, 'x']
            y_ap = df_at_time.loc[j, 'y']
            z_ap = df_at_time.loc[j, 'z']

            for k in range(len(cand_df_link)): 
                # 候補リンクからddr内のものを探し出す
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
                # 選択肢集合の中から適切な次の経路が見つからない場合，この過程がスキップさせるので結局辞書のvalueが全て0になる
                # 故にi（timestep）を更新する際にmax(dict~)が行えないので，そのタイミングで排除する
        
        # 辞書dict_for _each_timestepが全て0の場合，フォルダ分けて没にする？？？
        # 最初に二つフォルダ用意しておく
        #if all(value == 0 for value in dict_for_each_timestep.values()):
            #
        #    continue

        # 現時点でのresultをcsvにしてfailedフォルダに追加．名前はIDのを組み込む 

        # 最もvalueの大きいkeyを返す．それがestimated_linkになる
        estimeted_linkid = max(dict_for_each_timestep, key=dict_for_each_timestep.get) 
        each_result.append(estimeted_linkid) 
        next_chice_set = link_candidate(df_link, estimeted_linkid)
        C.append(next_chice_set)

    # ここまでで各userの経路パスが確定される，つまりeach_resultが出来上がる
    # result.append(each_result)
    # CSVファイルにデータを書き込む
    with open(f'/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_18/estimated_route/result_{file_name}', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(each_result)
        # for timestep, estimeted_linkid in enumerate(each_result, start=1):
            # writer.writerows(result)

    print(f"result_{file_name} に保存しました")

#resultはlistのlistになっている．これをdataframe形式のcsvにする
# print(result)