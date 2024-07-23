##### 観測モデル #####
import pandas as pd 
import networkx as nx
import os
import datetime
import numpy as np
from datetime import timedelta 
import csv
import time

start_time = time.time()
print('start!')

"""
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
"""
# DDR(RSSI)とweight(RSSI)．いづれはパラメタ入れる．暫定
# 早く関数にパラメタ入れて推定しろ＝＝＝＝＝
def ddr(rssi):
    return (rssi+10)/(-1*4)
def weight(rssi):
    return (rssi)/(-1*50)



# reading NW data
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv")

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/selected" # 1000データのみに対して実施（それで十分）
file_list = list(os.listdir(search_folder))
print(f'{len(file_list)}読み込みました') 
count = 0 # 出力ファイル数

output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17"
# if not os.path.exists(output_folder):
#     os.mkdir(output_folder)

# 結果格納用のファイル
df_result_tot = pd.DataFrame(columns=['user_id'] + ['t'] + [str(i) for i in range(1, 100) if i % 2 != 0])
# 初期化時点ではNaNで埋められている

# route estimation
for file_name in file_list: # userごとの処理
    if file_name.endswith('.csv'): # DS_Storeファイルの読み込み回避
        # ファイルの読み込み
        file_path = os.path.join(search_folder, file_name)
        df = pd.read_csv(file_path) 

        # time_step（10秒間隔）
        df['TIME'] = pd.to_datetime(df['TIME']) # datetime形式に変換
        time_delta = 10                         
        start_time = df.loc[0, 'TIME']
        for i in range(len(df)):
            time_diff = (df.loc[i, 'TIME'] - start_time).total_seconds()
            df.loc[i, 'time_step'] = time_diff // (time_delta) +1
        
        # time_stepが2以下の場合，処理せず次のfile_nameに移行
        if (df.loc[len(df)-1, 'time_step'] == 1) or (df.loc[len(df)-1, 'time_step'] == 2):
            continue

        # grouping by 'time_step'
        grouped = df.groupby('time_step') # groupbyメソッドで分割してもindexは保たれる

        # 各time_stepごとにデータ分割しリスト化
        dataframes_list = [group.reset_index(drop=True) for name, group in grouped] # indexリセット
        num_timestep = len(dataframes_list) # timestep数

        # numpy_totの初期化
        numpy_tot = np.zeros(len(df_link)+2) # 最後にuser_id列とtimestep列が先頭に来るので長さを+2している

        # 各time_stepにおけるリンクの観測確率の算定
        for i in range(len(dataframes_list)):  
            # time_step内に含まれる個別ログデータの集合
            df_at_time = dataframes_list[i]

            # time_stepスタンプ
            time_step = df_at_time.loc[0, 'time_step']

            # numpyの初期化．後でnumpy_totに結合
            numpy = np.zeros(len(df_link))

            for j in range(len(df_at_time)):
                # timestep内の個々のデータ
                rssi = df_at_time.loc[j, 'RSSI']

                # likelihood配列の初期化
                likelihood = np.zeros(len(df_link))

                # 各データiの検出bleの座標を取得
                x_ap = df_at_time.loc[j, 'x']
                y_ap = df_at_time.loc[j, 'y']
                z_ap = df_at_time.loc[j, 'floor']*1000 
                # zの値をめちゃ大きくすることで，階を挟んだ電波捕捉を捨象（階の厳密な高さが不明なのと，床板・天井版による電波遮蔽を考慮するため）．階が同じなら0になるので変な影響はない

                for k in range(len(df_link)):
                    linkid = df_link.loc[k, 'link_id']
                    index = linkid // 2 # linkidは奇数なので列はlinkid//2+1列目．indexは列数-1なので結局linkid // 2

                    x_o = df_link.loc[k, 'x_o']
                    y_o = df_link.loc[k, 'y_o']
                    z_o = df_link.loc[k, 'z_o']*1000

                    x_d = df_link.loc[k, 'x_d']
                    y_d = df_link.loc[k, 'y_d']
                    z_d = df_link.loc[k, 'z_d']*1000

                    x_mid = df_link.loc[k, 'x_mid']
                    y_mid = df_link.loc[k, 'y_mid']
                    z_mid = df_link.loc[k, 'z_mid']*1000
                
                    # 1/4点
                    x_qu = x_o/4 + x_d/4*3
                    y_qu = y_o/4 + y_d/4*3
                    z_qu = (z_o/4 + z_d/4*3) * 1000

                    # 3/4点
                    x_qua = x_o/4*3 + x_d/4
                    y_qua = y_o/4*3 + y_d/4
                    z_qua = (z_o/4*3 + z_d/4) * 1000

                    # 距離を5種類計算
                    d_mid = ((x_mid - x_ap)**2 + (y_mid - y_ap)**2 + (z_mid - z_ap)**2)**0.5
                    d_o = ((x_o - x_ap)**2 + (y_o - y_ap)**2 + (z_o - z_ap)**2)**0.5
                    d_d = ((x_d - x_ap)**2 + (y_d - y_ap)**2 + (z_d - z_ap)**2)**0.5
                    d_qu = ((x_qu - x_ap)**2 + (y_qu - y_ap)**2 + (z_qu - z_ap)**2)**0.5
                    d_qua = ((x_qua - x_ap)**2 + (y_qua - y_ap)**2 + (z_qua - z_ap)**2)**0.5

                    # 5種類のうち最小のものがddrより内側にあれば「観測」判定
                    d = min(d_mid, d_o, d_d, d_qu, d_qua)
                    if d < ddr(rssi):
                        likelihood[index] = 1 # DDR内判定されたリンクのところが1になる

                # DDR内のリンクに対してrssiに応じた重みづけ．1になっているところだけ重みづけされるので丁度良い
                likelihood = likelihood * weight(rssi)

                """
                # 各リンクの尤度はDDR内のリンク数をsumsとして1/sumsとする
                sums = likelihood.sum()
                if sums == 0: # ddrが小さすぎる＝強度が大きすぎるとリンクが捕捉されなくてsums=0になる
                    # continueにすると以下の処理がスキップされるのでnumpyが作られずnumpyがnunになってしまう!numpyの中身はintであることが想定されているので後の処理でエラーになる
                    likelihood = likelihood # もうしょうがないのでこの時は尤度0にする
                else:
                    likelihood = likelihood/sums
                """

                # time_step内で確率を出すので，個々のデータで得られた重みづけの結果をループが移行する前に足しておく（行として追加）
                numpy = np.vstack((numpy, likelihood))

                ## ここまでで各個別のログデータに対する処理が終了→次のログデータへ
                
            # timestep内の各重みを合計する，つまりnumpyの各列axis=0の和を取る．numpy[i]がtime_step i における各linkの重みの総和
            column_sums = np.sum(numpy, axis=0)

            # timestep内で割り当てられた全重みの総和
            total_weight = np.sum(column_sums)

            # timestep内での各リンクの確率を計算
            link_probability = column_sums/total_weight

            # 要素数は1．timestepの値をlink_probabilityの先頭に入れる．そのためにはnumpy配列化してから挿入する必要がある
            time_step_array = np.array([time_step])

            # time_stepをlink_probabilityの先頭に追加．これがtimestep tでの観測結果となる
            t_result = np.concatenate((time_step_array, link_probability))

            # user idを記録（後で時間帯毎に一つのcsvにまとめるための処理
            base_name = file_name.split('.')[0] # 数字部分のみ取り出す
            user_id = int(base_name) # int型に
            user_id_array = np.array([user_id])
            t_result_withid = np.concatenate((user_id_array, t_result))

            # numpy_tot にtimestep i の尤度情報を付け足し
            numpy_tot = np.vstack((numpy_tot, t_result_withid))
        
        # ここまでで全timestepに対して結果が出て，numpy_totに結果が入ってる
        # numpy_totの初期設定で入れておいた0の列を消去
        numpy_tot = np.delete(numpy_tot, 0, axis=0)
        
        # csvファイル化を見据え，numpyをdataframe化
        df_result = pd.DataFrame(data=numpy_tot)

        # columnsを設定．第0列目はtなので，link尤度の情報が入ってるのは第1~51列
        df_result.columns = ['user_id'] + ['t'] + [str(i) for i in range(1, 100) if i % 2 != 0]

        # o, d（それぞれtimestepの先頭・最後尾で尤度が最大であったもの）を取得．1:51で1~51列目までを検索する
        # 確率最大リンクが複数ある場合，idxmaxメソッドは見つかった最初のindexを返す．linkの設定で階段は後ろの方に設定しておいたので，コンコースかホーム上のリンクが優先される．よって階段とホームの尤度が最大であったときもホームを採用するから問題なし
        o_link = df_result.iloc[0, 2:53].idxmax() 
        d_link = df_result.iloc[num_timestep-1, 2:53].idxmax()

        # o_link, d_linkは列名なのでstr．intに変換
        o_link = int(o_link)
        d_link = int(d_link)

        # o_linkあるいはd_linkが階段リンク（71-99の奇数）だった場合，to_csvをしないでループを飛ばす
        stairs = [i for i in range(70, 100) if i % 2 != 0]
        if (o_link in stairs) or (d_link in stairs):
            continue
        
        df_result['o_link'] = o_link
        df_result['d_link'] = d_link

        # result_path = os.path.join(output_folder, f"{file_name}")
        # df_result.to_csv(result_path)

        count += 1
        # ここまでであるuserに対する処理が終了

        df_result_tot = pd.concat([df_result_tot, df_result], ignore_index=True)

result_path = os.path.join(output_folder, "measuring_result.csv")
df_result_tot.to_csv(result_path)

print(f'{count}ユーザの観測結果が出力されました')