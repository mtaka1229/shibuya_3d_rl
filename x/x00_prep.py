import pandas as pd 
import datetime
import numpy as np 
import os

# 時間帯別の親データに対する処理
###################################################################
########## 混雑度densityの計算とデータ付与，および不要列カット ###########
###################################################################
# df_parents = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_17.csv')

### 不要列の排除と15分間のみのデータ抽出 ### 
"""
df_parents = df_parents.iloc[:, 2:]

# 不要列排除
df_parents.drop(['num', 'type', 'day', 'hour', 'day_flag'], axis=1, inplace=True)

df_parents['ID'] = df_parents['ID'].str.replace('id', '').astype(int)
df_parents['time'] = pd.to_datetime(df_parents['time'])

df_parents.to_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_17.csv')
print(df_parents)

# 17:00-17:15のデータを取り出す．あまりに重いので．
df['time'] = pd.to_datetime(df['time'])
# cutline = datetime.datetime(2022, 12, 18, 17, 0, 0)
# Timestampはdatetimeをpandas上でも機能させるように作られたもの
cutline = pd.Timestamp('2022-12-18 17:15:00+09:00')

#print(df.loc[1, 'time'].dtype)
df_quater = df[df['time'] < cutline]
df_quater.to_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_1715.csv')
print(df_quater)
#print(df.dtypes)
"""
### 不要列の排除と抽出完了 ### # dfは17:00-17:15までのデータ．綺麗に1/4になったが，それでもレコード数は100万超

# 10sec間隔でtimestepを与える（timestep列）
df_quater = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_1715.csv')

# print(df_quater.dtypes) 最初timeはobjectなのでdatetime形式に変換
df_quater['time'] = pd.to_datetime(df_quater['time'])

# timeの昇順でデータを並べ替え
df_quater = df_quater.sort_values('time', ascending=True).reset_index(drop=True)

# time_stepを10秒間隔で付与
time_delta = pd.Timedelta(seconds=10)
df_quater['timestep'] = (df_quater['time'] - df_quater['time'].iloc[0]) // time_delta + 1

#### Cmt計算 ####
# 混雑度記録用のmatrix（beacon数23*timestep数90）c[m, t]はbeacon mのtimestep tにおけるユニークmacアドレス数
Cmt = np.zeros((23, 90), dtype = int)

# bleビーコンごとにgroup化してリスト化（各dataframe内でindexもリセット）
grouped_beacon = df_quater.groupby('ID')
# beacon_list = [group.reset_index(drop=True) for name, group in grouped_beacon]
beacon_list = []
for group_name, group_data in grouped_beacon:
    group_data = group_data.reset_index(drop=True)
    beacon_list.append(group_data)

for beacon_data in beacon_list:
    # beaconの番号を得る
    # print(beacon_data)
    beacon_id = int(beacon_data.loc[0, 'ID'])
    grouped_beacon_timestep = beacon_data.groupby('timestep')

    beacon_timestep_list = []
    # beacon_timestep_list = [group.reset_.index(drop=True) for name, group in grouped_beacon_timestep]
    for group_name, group_data in grouped_beacon_timestep:
        group_data = group_data.reset_index(drop=True)
        beacon_timestep_list.append(group_data)    

    for beacon_timestep_data in beacon_timestep_list:
        # timestepを得る
        timestep = int(beacon_timestep_data.loc[0, 'timestep'])
        unique_mac_count = beacon_timestep_data['MAC'].nunique()
        # Cmtにこの値を入れる
        Cmt[beacon_id-1, timestep-1] = unique_mac_count
#### Cmt完成 ####

#### mscアドレスごとにデータ出し直し ####
# groupbyでmacアドレスごとに分割して処理
grouped_mac = df_quater.groupby('MAC')

# グループ化されたDataFrameのリストを作成 # 各DataFrameのインデックスをリセット
mac_list = [group for name, group in grouped_mac]
mac_list = [mac_data.reset_index(drop=True) for mac_data in mac_list]
# mac_list = [group.reset_index(drop=True) for name, group in grouped] # これだと上手くいかなかった

output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

#### 1000userの抽出
count = 0
while count <= 1000:
    # 1000userのデータが取れたら終了（後でtimestepが飛んでいるやつも排除するが，このくらいあれば良いのでは）
    for mac_data in mac_list:
        # 全体でtimestep振っているのでtimestep列の最大値と最小値の差が2以下の場合，continue
        timestep_min = mac_data.loc[0, 'timestep']
        timestep_max = mac_data.iloc[-1, mac_data.columns.get_loc('timestep')]
        if timestep_max - timestep_min <= 2:
            continue
        
        # ホーム階or改札階のいずれかのビーコンで観測されていない場合，continue
        measured_beacons = mac_data['ID'].unique().tolist()
        # bleの階ごとのlist．工事の前中後で変化するので注意
        ble_at_plathome = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 22, 23] # "id001", "id002", "id003", "id004", "id005", "id006", "id008", "id009", "id010", "id011", "id022", "id023"]
        ble_at_concourse = [7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] #  "id007","id012", "id013", "id014", "id015", "id016", "id017", "id018", "id019", "id020", "id021"]
        if (not any(item in measured_beacons for item in ble_at_plathome)) or (not any(item in measured_beacons for item in ble_at_concourse)):
            continue
        
        # ここで混雑度列与える
        mac_data['dens'] = 0
        # timestepとbeaconを読んで，該当するCmtを参照して'dens'に与える
        for i in range(len(mac_data)): # user内の各データを参照して混雑度を付与．intもtimestepもintのはず
            timestep = mac_data.loc[i, 'timestep']
            beacon = mac_data.loc[i, 'ID']
            mac_data.loc[i, 'dens'] = Cmt[beacon-1, timestep-1]
        # ここはinputデータを作る段階なのでpyファイル一回区切ってOK

        # 諸条件満たしたら晴れてcsv化．iにはcountを用いれば良い．
        output_file = os.path.join(output_folder, f"{count}.csv")
        mac_data.to_csv(output_file, index=False)

        # 出力したらcount増やす
        count += 1  # なんか上手く機能しなかったみたいで2万以上出力されたが，かなり速かった．3分くらいだったのでまあ結果オーライということで．