# ぼつ
# 各日付の生データから渋谷の歩行データを抽出
import pandas as pd 
import numpy as np 
import os 
from datetime import datetime as dt 
import re

base_path = '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps'
base_dirs = ["new", "old"]
file_names = ["t_loc_data.csv", "t_trip.csv"]

##### PPデータの範囲を取得
"""
folder_pattern = re.compile(r'^10029_20\d{6}$')

max_lat = 0
min_lat = 300
max_lon = 0
min_lon = 300
for base_dir in base_dirs:
    full_base_dir = os.path.join(base_path, base_dir)  #  '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/new'
    for folder_name in os.listdir(full_base_dir): # folder_nameは10029_20\d{6}$'

        #if folder_name.endswith('.csv'):
        if folder_pattern.match(folder_name):
            folder_path = os.path.join(full_base_dir, folder_name) # '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/new/10029_20\d{6}$'
            file_path = os.path.join(folder_path, "t_loc_data.csv")
            print(file_path)
            df = pd.read_csv(file_path, encoding='shift-jis') # これでファイル読むはず
            print(df.head(10))
            for i in range(len(df)):
                lat = float(df.loc[i, '経度'])
                lon = float(df.loc[i, '緯度'])
                max_lat = max(lat, max_lat)
                min_lat = min(lat, min_lat)
                max_lon = max(lon, max_lon)
                min_lon = min(lon, min_lon)
print(max_lat, min_lat, max_lon, min_lon)
"""

#### 取得された範囲を2km四方のセルで分割する
import geopy.distance
import math
import csv

map_folder_path = '/Users/takahiromatsunaga/res2023/shibuya_nw/map'

# 長方形の4頂点の緯度経度
points = [
    (35.757701, 139.86511),    # (lat, lon) format
    (35.438085, 139.86511),
    (35.757701, 139.569292),
    (35.438085, 139.569292)
]
print(points)
# 緯度と経度の範囲を取得
min_lat = min(p[0] for p in points)
max_lat = max(p[0] for p in points)
min_lon = min(p[1] for p in points)
max_lon = max(p[1] for p in points)

# セルのサイズ（2km）
cell_size_km = 2.0

# セルサイズを度に変換（大まかな計算）
def km_to_deg(km, at_lat):
    lat_km_per_deg = 111.32
    lon_km_per_deg = 40075 * math.cos(math.radians(at_lat)) / 360
    return km / lat_km_per_deg, km / lon_km_per_deg

cell_size_lat, cell_size_lon = km_to_deg(cell_size_km, (min_lat + max_lat) / 2)

# グリッドの頂点を計算
lat = min_lat # latの初期値
grid_points = []
point_id = 1

print(point_id)

## ここの処理が遅い
while lat <= max_lat + cell_size_lat:
    lon = min_lon
    while lon <= max_lon + cell_size_lon:
        grid_points.append((point_id, lat, lon))
        lon += cell_size_lon
        point_id += 1
    lat += cell_size_lat

# 最大値に合わせて格子を生成
# while lat >= min_lat - cell_size_lat:
#     lon = max_lon
#     while lon >= min_lon - cell_size_lon:
#         grid_points.append((point_id, lon, lat))
#         point_id += 1
#         lon -= cell_size_lon
#     lat -= cell_size_lat

# Generate cells from grid points
cells = []
cell_id = 1
num_cols = int((max_lon - min_lon) / cell_size_lon) + 1

print(cells)

for i in range(len(grid_points) - num_cols - 1):
    if (i + 1) % num_cols == 0:
        continue  # Skip the last point in each row
    id1 = grid_points[i][0]
    id2 = grid_points[i + 1][0]
    id3 = grid_points[i + num_cols][0]
    id4 = grid_points[i + num_cols + 1][0]
    cells.append((cell_id, id1, id2, id3, id4))
    cell_id += 1

# Save to CSV
cell_filename = 'cells.csv'

print(cell_filename)

with open(os.path.join(map_folder_path, cell_filename), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cell_id', 'id1', 'id2', 'id3', 'id4'])
    for cell in cells:
        writer.writerow(cell)



gridpoint_filename = 'grid_points2.csv'

print(gridpoint_filename)

with open(os.path.join(map_folder_path, gridpoint_filename), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['gridpoint_id', 'lon', 'lat'])
    for point in grid_points:
        writer.writerow(point)

# グリッドポイントを出力
# for point in grid_points:
#     print(f"Latitude: {point[0]}, Longitude: {point[1]}")






# # # 結果を表示
# # for file_name, paths in found_files.items():
# #     print(f"{file_name} found in:")
# #     for path in paths:
# #         print(f"  - {path}")

# # df_feederが各日のt_locfeeder, df_locがt_locdataとする．t_tripデータは使用しない（目的を使わなければ）
# # df_tripをt_tripデータとして目的コードを抽出する
# dict = {}
# # テスト
# df_feeder = pd.read_csv('/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/new/10029_20230110/t_locfeeder.csv', encoding='shift-jis')
# df_loc = pd.read_csv('/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/new/10029_20230110/t_loc_data.csv', encoding='shift-jis')
# df_trip = pd.read_csv('/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/new/10029_20230110/t_trip.csv', encoding='shift-jis')

# for i in range(len(df_feeder)):
#     if df_feeder.loc[i, '移動手段コード'] == 100:
#         # then
#         value_list = []
#         tripid = int(df_feeder.loc[i, 'ユーザーID'])
#         userid = int(df_feeder.loc[i, 'トリップID'])
#         print(tripid, userid)

#         # もしすでにtripid が登場していたら
#         if tripid in dict:
#             # tripidがkeyのvalueのうち2番目
#             in_time = dict[tripid][1] # 既往in時刻
#             in_time = min(in_time, dt.strptime(df_feeder.loc[i, '記録日時'], "%Y-%m-%d %H:%M:%S"))
#             out_time = dict[tripid][2] # 既往out時刻
#             out_time = max(out_time, dt.strptime(df_feeder.loc[i, '作成日時'], "%Y-%m-%d %H:%M:%S"))
#             # この内容でin_time, out_time更新．目的はまあ同じでいいか．
#             value_list = dict[tripid]
#             # value_list[1]がin_time, [2]がout_time
#             value_list[1] = in_time
#             value_list[2] = out_time
#         else:
#             # 時間形式を直す
#             # in_time = dt.datetime(df_feeder.loc[i, '記録日時'])
#             in_time = dt.strptime(df_feeder.loc[i, '記録日時'], "%Y-%m-%d %H:%M:%S")
#             # out_time = dt.datetime(df_feeder.loc[i, '作成日時'])
#             out_time = dt.strptime(df_feeder.loc[i, '作成日時'], "%Y-%m-%d %H:%M:%S")
#             print(in_time, out_time)
#             # ここで目的コードも回収しとくか．．
#             print(df_trip)
#             # ここまではOK
            
#             print(df_trip[df_trip['ID']== tripid]['目的コード（active）'].iloc[0]) # iloc[0]をつけないとseriesを処理できない
#             purpose = int(df_trip[df_trip['ID']== tripid]['目的コード（active）'].iloc[0])
#             value_list.append(userid)
#             value_list.append(in_time)
#             value_list.append(out_time)
#             value_list.append(purpose)
#             dict[tripid] = value_list

#     else: # 徒歩でないトリップは無視
#         continue
# # ここでdictにトリップidと時間が入ったはず
# print(dict)


# # loc_dataをユーザごとにgroupbyする
# grouped = df_loc.groupby(['ユーザーID']) # listにして渡す必要がある
# df_loc_list = [group.reset_index(drop=True) for name, group in grouped]
# for group in df_loc_list: # groupはdataframe構造





# 
# これでfound_filesにアイテム入った
# まず，各ユーザの各unlinkedトリップに対して徒歩の移動の開始時刻と終了時刻を取得．
# 
# 1ユーザで複数の徒歩があった場合は複数のまとまりで記録
# 
# 渋谷の範囲は35.666-35.652，139.694-139.710
# この範囲に入ってて，かつ
