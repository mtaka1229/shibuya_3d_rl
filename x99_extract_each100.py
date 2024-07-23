import pandas as pd
import os

"""
search_folder = "/Users/takahiromatsunaga/bledata/"
new_folder = "/Users/takahiromatsunaga/bledata_reduced/"

# 新しいディレクトリが存在しない場合は作成する
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# search_folder内の各CSVファイルに対して処理を行う
for csv_file in os.listdir(search_folder):
    file_path = os.path.join(search_folder,csv_file)

    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    
    #上から100個抜き出した方が計算量少ない
    #df = df.head(100)
    # データを1/100に縮小する（先頭から100行ごとに1行をサンプリング）つまり各時間帯でサンプルが一応検出されることになる．
    df = df.iloc[::100, :]

    # 新しいファイル名を生成する．.csvのところを_reduced追加して置き換えている
    new_csv_file = csv_file.replace(".csv", "_reduced.csv")
    new_file_path = os.path.join(new_folder, new_csv_file)

    # 縮小されたデータを新しいディレクトリに保存する
    df.to_csv(new_file_path, index=False)

    # 処理が完了したらメモリを解放する（省略可能）
    del df

print("処理が完了しました。")
"""

from datetime import datetime, timedelta

search_folder = "/Users/takahiromatsunaga/bledata_reduced/"
for file_name in os.listdir(search_folder):
    file_path = os.path.join(search_folder,file_name)
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])

    df['time_only'] = df['time'].apply(lambda x: x.strftime('%H:%M:%S'))

    """"
    # 時刻の範囲を設定する
    start_time = datetime.strptime('2022-12-18 13:30:00', '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime('2022-12-18 13:40:00', '%Y-%m-%d %H:%M:%S')

    # 条件を指定して行をフィルタリングする
    filtered_df = df[(df['時刻'] >= start_time) & (df['時刻'] <= end_time)]

    # フィルタリングされたデータを新しいCSVファイルとして保存する
    filtered_df.to_csv('新しいファイル.csv', index=False)
    """

    