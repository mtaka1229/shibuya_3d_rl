# 経路推定の下準備(生データに対して）
# 生データに観測されたbleビーコンの座標をくっつける
import pandas as pd 
import os 

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/alladdress"
# とりあえず2022/12/18は08も17も終了

# ble_nwの方を作り替えた（id013が2Fになっていた）ので，この20221218_08, 17, 20221219_08に対して作業やり直し
# その前に以前に作ったx, y, floor列を削除
# 20221219_08以外は処理終了．20221219_08はちょっと厄介だったので0から作り直す
"""
file_list = list(os.listdir(search_folder))
for file_name in file_list:
    if file_name.endswith(".csv"): # DS_Store回避のための行
        file_path = os.path.join(search_folder, file_name)
        df = pd.read_csv(file_path)

        if 'x' in df.columns: 
            # x, y, floor列を削除→num列までを保存
            # x列のindex
            x_index = df.columns.get_loc('x')

            df = df.iloc[:, :x_index] # x列以降を削除
            # df = df.iloc[:, ID_column_index:] # 参考

            df.to_csv(file_path) # これで更新
print('finished!')

"""

# id_numと座標の対応を示したデータの読み込み
# x, y, floor列が追加される
ble_loc = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/ble_nw.csv") # id013を修正済み

file_list = list(os.listdir(search_folder))
for file_name in file_list:
    if file_name.endswith(".csv"): # DS_Store回避のための行
        file_path = os.path.join(search_folder,file_name)
        df = pd.read_csv(file_path)
        # ble_locをid_numをキーにしてマージ（結合）
        df = df.merge(ble_loc, on='id_num', how='left')
        df.to_csv(file_path)

### .DS_Storeを除去すれば良い！！！