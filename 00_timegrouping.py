#########################################################################
######### 生データから特定の日の特定の時間のデータを抜き出すためのコード ##########
#########################################################################

import pandas as pd 
import os

# 生データのフォルダを指定
search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_raw"
time_stamp = '20230107'
hour_stamp = '08'
# 最初にデータフレームを新規で設定
new_df = pd.DataFrame()

# 日付を指定
for file_name in os.listdir(search_folder):
    if file_name.endswith(f"{time_stamp}.csv"): # 全ビーコンについて，timestampが指定する日付のデータを参照 # 日付指定もそうだが，これによってDS_Storeファイルを回避しているので非常に重要！！
        file_path = os.path.join(search_folder,file_name)
        df = pd.read_csv(file_path)
    
        # 時刻を指定
        new_df = pd.concat([new_df, df[df["hour"] == int(hour_stamp)]], ignore_index=True)

# outputフォルダを指定・作成
output_folder = f"/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/{time_stamp}_{hour_stamp}"
os.makedirs(output_folder, exist_ok=True)

output_file = f'{time_stamp}_{hour_stamp}.csv'
# 一応出力しておく
#new_df.to_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/20221219_08.csv")
new_df.to_csv(os.path.join(output_folder, output_file))