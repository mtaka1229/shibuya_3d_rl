#########################################################################
######### time_groupingで抜き出したデータからmacアドレス毎のcsvを作るためのコード ##########
#########################################################################

import os
import pandas as pd

# 01で日時を特定した全データを出力済みなので，それを読み込む
df = pd.read_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/20221219_08.csv")

# outputフォルダの指定・作成
output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/alladdress"
os.mkdir(output_folder)

# MACアドレスごとにファイルを分割して保存
# unique関数
for i, mac_address in enumerate(df["MAC"].unique(), start=1):
    mac_df = df[df["MAC"] == mac_address]

    # ここでindexをファイル名にする
    output_file = os.path.join(output_folder, f"{i}.csv")
    mac_df.to_csv(output_file, index=False)
    

# 20221219_08は4万ちょいで打ち切り