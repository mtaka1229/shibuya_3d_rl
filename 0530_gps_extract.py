## locデータから渋谷駅まち回遊部分をユーザごとに抽出しcsv化するコード
import pandas as pd 
import numpy as np 
import os 
from datetime import datetime as dt, timedelta
import math

base_path = '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps'
df_diary = pd.read_csv(os.path.join(base_path, 'activity_RL/input/diary.csv'))
df_choice = pd.read_csv(os.path.join(base_path, 'activity_RL/input/choice.csv'))

shibuya_sta_loc = (35.658999617226975, 139.7010586998745)

# 緯度経度から距離を計算する関数
def Dis(lon_error, lat_error):
    R = 6378137
    THETA = 33.85/180 * math.pi
    lon_1 = 2*math.pi * R * math.cos(THETA) / 360 # 経度1度あたりの距離を計算しておく
    lat_1 = 2 * math.pi * R / 360 # 緯度1度あたりの距離を計算しておく
    # lon_error, lat_errorにそれぞれlon_1, lat_1を掛けて各水平距離を計算
    lon_dis = lon_error * lon_1
    lat_dis = lat_error * lat_1
    dis = lon_dis ** 2 + lat_dis ** 2
    return dis ** 0.5

# df_feederが各日のt_locfeeder, df_locがt_locdataとする．t_tripデータは使用しない（目的を使わなければ）
# df_tripをt_tripデータとして目的コードを抽出する
def gps_df_reader(date, newold):
    if newold == 'new':
        s = 10029
    elif newold == 'old':
        s = 10028
    elif newold == '202311': 
        s = 10032
    date = date.replace('-', '')
    df_feeder = pd.read_csv(os.path.join(base_path, f'{newold}/{s}_{date}/t_locfeeder.csv'), encoding='shift-jis')
    df_loc = pd.read_csv(os.path.join(base_path, f'{newold}/{s}_{date}/t_loc_data.csv'), encoding='shift-jis')
    df_trip = pd.read_csv(os.path.join(base_path, f'{newold}/{s}_{date}/t_trip.csv'), encoding='shift-jis')
    return {'feeder': df_feeder, 'loc': df_loc, 'trip': df_trip}

# まずdiaryデータから→1の移動を抽出
def to_shibuya_trip_extract():
    # list of (diaryNo, userid)で抽出
    indices = df_choice[(df_choice['destination'] == 1.0) & (df_choice['trip'] == 1.0)].index # 全然ない
    indices_list = set(indices) # 重複なしのはず
    diary_no_list = [i // 24 - 1 for i in indices_list] # これがdiary noのリストのはず
    return diary_no_list

def extract_userid_date(diary_no_list):
    userid_date_list = []
    for j in diary_no_list:
        matches = df_diary[df_diary['diaryNo'] == j][['userid', 'date']] # こいう書き方できるのか．．
        userid_date_list.extend(list(matches.itertuples(index=False, name=None)))
    return userid_date_list

# userid_date_list
def extract_gps(userid_date_list):
    dict_list = []
    df_filtered_list = []
    for userid, date in userid_date_list: # 別の日に同じユーザが出てきても別で扱う
        trip_dict = {}
        newold = 'old' if userid < 23251 else 'new'
        # print('userid', userid, 'date', date)
        # データ読み込み
        data = gps_df_reader(date, newold)
        df_feeder = data['feeder']
        df_loc = data['loc']
        df_trip = data['trip']

        # 該当する個人の特定日のデータのみ抜き出す
        df_feeder = df_feeder[df_feeder['ユーザーID'] == userid].reset_index(drop=True) ## これないと変になった！！
        df_loc = df_loc[df_loc['ユーザーID'] == userid].reset_index(drop=True)
        df_trip = df_trip[df_trip['ユーザーID'] == userid].reset_index(drop=True)
            
        for i in range(len(df_feeder)):
            if df_feeder.loc[i, '移動手段コード'] == 100: # 徒歩データの場合
                # then
                value_list = []
                tripid = int(df_feeder.loc[i, 'トリップID'])
                print(tripid, userid) # useridはすでに取得済み

                # もしすでにtripid が登場していたら
                if tripid in trip_dict:
                    # tripidがkeyのvalueのうち2番目
                    in_time = trip_dict[tripid][1] # 既往in時刻
                    in_time = min(in_time, pd.to_datetime(df_feeder.loc[i, '記録日時'], format="%Y-%m-%d %H:%M:%S"))
                    out_time = trip_dict[tripid][2] # 既往out時刻
                    out_time = max(out_time, pd.to_datetime(df_feeder.loc[i, '作成日時'], format="%Y-%m-%d %H:%M:%S"))
                    # この内容でin_time, out_time更新．目的はまあ同じでいいか．
                    value_list = trip_dict[tripid] # value_list[1]がin_time, [2]がout_time
                    value_list[1] = in_time
                    value_list[2] = out_time
                else:
                    in_time = pd.to_datetime(df_feeder.loc[i, '記録日時'], format="%Y-%m-%d %H:%M:%S")
                    out_time = pd.to_datetime(df_feeder.loc[i, '作成日時'], format="%Y-%m-%d %H:%M:%S")
                    #print(in_time, out_time)
                    #print(df_trip)
                    # ここまではOK
                    
                    # print(df_trip[df_trip['ID']== tripid]['目的コード（active）'].iloc[0]) # iloc[0]をつけないとseriesを処理できない
                    purpose = int(df_trip[df_trip['ID']== tripid]['目的コード（active）'].iloc[0])
                    trip_dict[tripid] = [userid, in_time, out_time, purpose]

        # ここでdictにトリップidと時間が入ったはず
        dict_list.append(trip_dict)
        # print(dict) # 確認

        # ここから各ユーザ日時のgpsデータを抽出する
        # in/out data
        # shibuya_sta_loc = (35.658999617226975, 139.7010586998745)
        filtered_loc_data = []

        for tripid, values in trip_dict.items():
            in_time, out_time = values[1], values[2]
            for j in range(len(df_loc)):
                record_time = pd.to_datetime(df_loc.loc[j, '記録日時'], format="%Y-%m-%d %H:%M:%S")
                if in_time <= record_time <= out_time:
                    lat_error = df_loc.loc[j, '緯度'] - shibuya_sta_loc[0]
                    lon_error = df_loc.loc[j, '経度'] - shibuya_sta_loc[1]
                    if Dis(lon_error, lat_error) <= 500:  # 1km以内→550m圏内
                        filtered_loc_data.append(df_loc.iloc[j])
        
        if filtered_loc_data:
            df_filtered = pd.DataFrame(filtered_loc_data)
            df_filtered_list.append(df_filtered)

    return dict_list, df_filtered_list # 渋谷駅1km圏内の（おそらく徒歩）locデータが濾過されるはず，．．


# ## 1繋がりのデータから渋谷→周辺部の歩行データのみ取り出す
def split_by_time_difference(df, time_column, threshold_seconds=20):
    # 時系列の差分を計算し、差分が閾値を超えた部分でデータを分割する
    df[time_column] = pd.to_datetime(df[time_column]) 
    df['time_diff'] = df[time_column].diff().abs()
    threshold = timedelta(seconds=threshold_seconds) # 前後で30sec以上の乖離がある時はデータを切る
    
    # 差分が閾値を超えるインデックスを取得
    split_indices = df.index[df['time_diff'] > threshold].tolist()
    split_indices.append(len(df))  # 最後のインデックスを追加
    
    # 分割されたデータフレームのリストを作成
    start_idx = 0
    split_dfs = []
    for idx in split_indices:
        split_dfs.append(df.iloc[start_idx:idx].drop(columns=['time_diff']))
        start_idx = idx
    
    return split_dfs


# test_folder_path = '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/shibuya_mezzo_gps'
# test_folders = os.listdir(test_folder_path)
# csv_list = []
# for file_name in test_folders:
#     if file_name.endswith('.csv'):
#         file_path = os.path.join(test_folder_path, file_name)
#         df = pd.read_csv(file_path, encoding='shift-jis')
#         csv_list.append(df)


def extract_from_shibuya(df_filtered_list):
    res = []
    print('入力ファイルの数', len(df_filtered_list))
    for dfi in df_filtered_list: # 個人のlocデータ
        if dfi.empty:
            print('dfi.empty!!!')
            continue
        
        # インデックスをリセット
        dfi = dfi.reset_index(drop=True)

        userid = dfi.loc[0, 'ユーザーID'] 
        date = dfi.loc[0, '記録日時'].split(' ')[0]
        date = date.replace('-', '')

        print('userid, date', userid, date)

        # 生成されたデータが複数あるとき，平均speedが5以下，渋谷駅から200m圏内のデータがあるもののみ残す
        # それでも複数ある時はデータ数が大きい方をとる
        # 時刻で分ける
        dfs_list = [] 
        sprit_dfs = split_by_time_difference(dfi, '記録日時', threshold_seconds=30)
        #print('スプリット長さ', len(sprit_dfs)) # 1なら分割できない

        dfs_dict = {}
        for sprit_df in sprit_dfs:
            # print(sprit_df)
            sprit_df = sprit_df.reset_index(drop=True)
            if len(sprit_df) <= 10:
                print('短くて無理！')
                continue # データ少なすぎると無理なので

            #print('meanspeed', sprit_df['speed'].mean())
            if sprit_df['speed'].mean() >= 5:
                print('meanspeedでcontinue')
                continue
            #print(f'len(sprit_df){len(sprit_df)}')
            
            for j in range(len(sprit_df)):
                # print(f'jは{j}')
               # print(sprit_df)
                dist_list = []
                #print('緯度' , sprit_df.loc[j, '緯度'])
                lat_error = sprit_df.loc[j, '緯度'] - shibuya_sta_loc[0]
                lon_error = sprit_df.loc[j, '経度'] - shibuya_sta_loc[1]
                dist_list.append(Dis(lon_error, lat_error))
            
            print(f'mindist{min(dist_list)}')
            if min(dist_list) >= 150:
                print('mindistでcontinue')
                continue
            
            dfs_list.append(sprit_df) 
            print('条件満たした')

        # print(f'userid{userid}ここまできた，dfsの長さは{len(dfs_list)}')

        # 最も長さの長いデータフレームを取り出す
        # この処理なくすか
        count = 0
        if dfs_list: 
            # longest_df = max(dfs_list, key=len)
            # return longest_df
            for dfs in dfs_list:
                # res.append(longest_df)
                res.append(dfs)
                # longest_df.to_csv(os.path.join(base_path, f'shibuya_mezzo_gps3/{userid}_{date}_filtered_loc_data.csv'), index=False, encoding='shift-jis')
                dfs.to_csv(os.path.join(base_path, f'shibuya_mezzo_gps4/{userid}_{date}_{count}_filtered_loc_data.csv'), index=False, encoding='shift-jis')
                count += 1
        else: # dfs_listがない場合＝全部条件満たさない
            print('ダメーー')
        
        print(f'userid{userid}完了！')
    
    return res


# 動作確認
diary_no_list = to_shibuya_trip_extract()
print('diary_no_list', diary_no_list)
print(len(diary_no_list))
userid_date_list = extract_userid_date(diary_no_list)
print('userid_date_list', userid_date_list)
_, df_filtered_list = extract_gps(userid_date_list)
longest_list = extract_from_shibuya(df_filtered_list)
