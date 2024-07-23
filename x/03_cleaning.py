###############################################################################
######### 02_address_separationで作ったアドレスごとのファイルをクリーニング ##########
###############################################################################

import pandas as pd 
import os
import datetime
import shutil

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20230107_08/alladdress"
trush_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20230107_08/trush"

if not os.path.exists(trush_folder):
    os.mkdir(trush_folder)

# bleの階ごとのlist作成. 工事の前中後で変化するので注意
ble_at_plathome = ["id001", "id002", "id003", "id004", "id005", "id006", "id008", "id009", "id010", "id011", "id022", "id023"]
ble_at_concourse = ["id007","id012", "id013", "id014", "id015", "id016", "id017", "id018", "id019", "id020", "id021"]

file_list = list(os.listdir(search_folder))
for file_name in file_list:
    file_path = os.path.join(search_folder, file_name)

    # 必殺奥義cp932
    df = pd.read_csv(file_path, encoding="cp932")

    # print(file_name) #['ID'])

    # ID列の中に入ってるbleのidを重複なくリストに入れる
    ble_id = df['ID'].unique().tolist()

    # ble_idのリストにble_at_plathome or ble_at_concourseの要素が入っていない場合，当該ファイルをゴミフォルダに移す
    if (not any(item in ble_id for item in ble_at_plathome)) or (not any(item in ble_id for item in ble_at_concourse)):
        shutil.move(file_path, trush_folder)

        
############# ここまででalladdressの中身はだいぶスッキリするはず #######################
############# 続いて残ったalladdressの中身を使える形で整理し直す #######################

# 更新された（はずの）new_search_folder_pathを再度読み込み，再度ファイル名も改める
for i, file_name in enumerate(os.listdir(search_folder)):
    file_path = os.path.join(search_folder, file_name)

    # 再度ここでindexをファイル名に
    new_file_name = f"{i}.csv"
    new_file_path = os.path.join(search_folder, new_file_name)

    # ファイルの名称を変更する
    os.rename(file_path, new_file_path)

    # データの読み込み
    df = pd.read_csv(new_file_path)

    # time列のデータをdatetime型に変換してTIME列に格納
    df['TIME'] = pd.to_datetime(df['time'])

    # ID列からidをとって数字にする「id_num」列
    df['id_num'] = df['ID'].str.replace('id', '').astype(int)

    # TIME列の昇順によって，全データを並べ替える
    df = df.sort_values('TIME', ascending=True).reset_index(drop=True)

    # きれいにしたものを再度保存
    df.to_csv(new_file_path)

    # ここまででデータが取れているものだけが残り，かつ利用しやすい形に直されているはず
