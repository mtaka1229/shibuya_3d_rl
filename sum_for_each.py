##############address_separationで抽出したそれぞれのmacアドレスのデータに対して，1分ごとにどのBLE機器で何回検出されたかの和を計算###################

import pandas as pd 
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import shutil

search_folder = "/Users/takahiromatsunaga/bledata/blescan_20221219_18"

#################まずは不要なデータを消去する######################
#################この部分がうまくいかない．他はうまくいく####################

new_search_folder_path = "/Users/takahiromatsunaga/bledata/blescan_20221219_18_cleaned"

#ここもうやったので20221219_08に関しては飛ばす
if not os.path.exists(new_search_folder_path):
    os.mkdir(new_search_folder_path)

file_list = list(os.listdir(search_folder))
for file in file_list:
    file_path = os.path.join(search_folder, file)  # ファイルのパスを作成
    if file.endswith(".csv"):
        #適切なファイルを新規_cleanedフォルダに移す
        shutil.move(file_path, new_search_folder_path)

#bleの階ごとのlist作成．時期，つまりsearch_folderの指定する日付に応じて変化するので注意
ble_at_plathome = ["id001", "id002", "id003", "id004", "id005", "id006", "id008", "id009", "id010", "id011", "id022", "id023"]
ble_at_concourse = ["id007", "id012", "id013", "id014", "id015", "id016", "id017", "id018", "id019", "id020", "id021"]

for file_name in os.listdir(new_search_folder_path):
    file_path = os.path.join(new_search_folder_path, file_name)

    #必殺奥義cp932
    df = pd.read_csv(file_path, encoding="cp932")

    #print(file_name) #['ID'])

    # ID列の中に入ってるbleのidを重複なくリストに入れる
    ble_id = df['ID'].unique().tolist()

    # ble_idのリストにble_at_plathomeの要素が入っていない，もしくはble_at_concourseの要素が入っていない場合，当該csvは消去
    if (not any(item in ble_id for item in ble_at_plathome)) or (not any(item in ble_id for item in ble_at_concourse)):
        os.remove(file_path)
        #print("引っかかりました")
        
#############ここまででnew_search_folder_pathの中身はだいぶスッキリするはず（およそ半減することを期待）#######################
#############続いてmacアドレスごとに各時間・各検出器での検出回数の和を取り，新規のcsvを作成する########################

#出力したデータフレームcsvを入れるためのfolderを作成and指定
new_folder = "/Users/takahiromatsunaga/bledata/blescan_20221219_18_agg"
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 更新された（はずの）new_search_folder_pathを再度読み込み，ファイル名も改める
for i, file_name in enumerate(os.listdir(new_search_folder_path)):
    file_path = os.path.join(new_search_folder_path, file_name)
    new_file_name = f"20221219_08_{i}.csv"
    new_file_path = os.path.join(new_search_folder_path, new_file_name)

    # ファイルの名称を変更する
    os.rename(file_path, new_file_path)

    # データの読み込み
    df = pd.read_csv(new_file_path)

    #time列のデータをdatetime型に変換してTIME列に格納
    df['TIME'] = pd.to_datetime(df['time'])

    #ID列からidをとって数字にする「id_num」列
    df['id_num'] = df['ID'].str.replace('id', '').astype(int)

    #TIME列の昇順によって，全データを並べ替える
    df = df.sort_values('TIME', ascending=True).reset_index(drop=True)

    #きれいにしたものを再度保存
    df.to_csv(new_file_path)

    #データフレームの設定
    columns = ["id001", "id002", "id003", "id004", "id005", "id006", "id008", "id009", "id010", "id011", "id022", "id023", "id012", "id013", "id014", "id015", "id016", "id017", "id018", "id007", "id019", "id020", "id021"]
    
    #countしたデータを入れるための空のデータフレームcsv
    df_new = pd.DataFrame(columns=columns)

    #平均rssiを出すための空のデータフレームcsv
    df_rssi = pd.DataFrame(columns=columns)

    #ここでスタートタイムを切り捨て，エンドタイムを切り上げたい，その上で1分おきに集計したい
    start_time = df.loc[0, 'TIME'].replace(second=0, microsecond=0)
    end_time = df.loc[df.index[-1], "TIME"].replace(second=0, microsecond=0)
    end_time = end_time + datetime.timedelta(minutes=1)

    # 1分ごとのタイムステップ
    delta = datetime.timedelta(minutes=1)

    # 最初の時間だけはdf_newに行を追加しておく
    df_new.loc[start_time] = 0
    df_rssi.loc[start_time] = 0

    current_time = start_time

    # while current_time <= end_time:←while文のせいでめちゃ無限ループエラーで大変だったので使い方には気をつける
    for i in range(len(df)):
        index_time = df.loc[i, "TIME"]
        id_value = df.loc[i, 'ID']

        # TIME列の昇順に並べ替えてる前提があるので上から抑えるだけでOK
        if index_time < current_time + delta:
                
            if current_time in df_new.index:
                df_new.loc[current_time, id_value] += 1

            else:
                df_new.loc[current_time] = 0
                df_new.loc[current_time, id_value] = 1

        else:
            current_time += delta  # 現在の時間を更新
            df_new.loc[current_time] = 0
            df_new.loc[current_time, id_value] = 1

    # macアドレスごとに作ったdf_newをnew_folderの直下に保存する．
    df_new_name = new_file_name.replace(".csv", "_count.csv")
    df_new_file_path = os.path.join(new_folder, df_new_name)

    # csvに出力
    df_new.to_csv(df_new_file_path)

    #count = 0 #カウンターの初期値0

    """
    ######################################続いてrssiのファイルを作る############################################################
    # while current_time <= end_time:←while文のせいでめちゃ無限ループエラーで大変だったので使い方には気をつける
    for i in range(len(df)):
        index_time = df.loc[i, "TIME"]
        id_value = df.loc[i, 'ID']
        rssi = df.loc[i, 'RSSI']

        # TIME列の昇順に並べ替えてる前提があるので上から抑えるだけでOK
        if index_time < current_time + delta: #当該time step内であれば
                
            if current_time in df_rssi.index:
                df_rssi.loc[current_time, id_value] += abs(rssi)  #違うかも．負なので絶対値取るべき．
                #count += 1 #カウンター1増やす

            else:
                df_rssi.loc[current_time] = 0
                #count = 1 #current_timeの切り替わりと同時にカウンターは0になるので，基本は行の追加と同時にcountを初期化してやれば良い
                df_rssi.loc[current_time, id_value] = abs(rssi)

        else:
            #直前で行いたい処理はここに書く
            #df_newの結果を使って平均を出せば良い
            if df_rssi.loc[current_time, id_value] != 0: #rssi!=0なら検出回数も非零なはず
                df_rssi.loc[current_time, id_value] = df_rssi.loc[current_time, id_value] / df_new.loc[current_time, id_value]

            current_time += delta  # 現在の時間を更新
            #count = 1
            df_rssi.loc[current_time] = 0
            df_rssi.loc[current_time, id_value] = abs(rssi)

    # macアドレスごとに作ったdf_newをnew_folderの直下に保存する．
    df_rssi_name = new_file_name.replace(".csv", "_rssi.csv")
    df_rssi_file_path = os.path.join(new_folder, df_rssi_name)

    # csvに出力
    df_rssi.to_csv(df_rssi_file_path)

    """
    
    ##################################ここまで################################################################

    ############macユーザごとのグラフを描画する段階##################
    #X = [i+1 for i in range(len(df_new_t.columns))]

    ###########################まずはcountするグラフを描画#################################
    
    df_new_t = df_new.T
    X = [i+1 for i in range(len(df_new_t.columns))]


    #（棒グラフが生える）横軸の本数を指定．これはidの数
    fig, axes = plt.subplots(len(df_new_t)-1) #-1しないとラベルの列の分だけ列数が追加されてしまう

    for ax in axes[:-1]:
        ax.tick_params(bottom=False, labelbottom=False)  # 一番下のax以外，x軸目盛を消す
    for i in range(len(axes)):
        ax = axes[i]
        row_data = df_new_t.iloc[i+1]
        Y = row_data.tolist()
        ax.bar(X, Y)  # 棒グラフ
        ax.set_yticks([10]) # y軸の目盛りを指定
        ax.set_ylim(0, 20)  # y軸の上限・下限を指定

    plt.subplots_adjust(wspace=0, hspace=0)  # 間を詰める
    plt.xlabel('time step')
    plt.ylabel('count at each BLE')

    #描画したグラフをpng形式で保存
    image_name = new_file_name.replace(".csv", "_count.png") # 要検討
    image_path = os.path.join(new_folder, image_name)
    plt.savefig(image_path, format="png", dpi=300)

    plt.close(fig)


########################続いて1分間の平均RSSIを縦軸とする##############################

"""
    df_rssi_t = df_rssi.T

    #（棒グラフが生える）横軸の本数を指定．これはidの数
    fig2, axes2 = plt.subplots(len(df_rssi_t)) #-1しないとラベルの列の分だけ列数が追加されてしまう(仮に1を引かないでやってみる）)

    for ax in axes2[:-1]:
        ax.tick_params(bottom=False, labelbottom=False)  # 一番下のax以外，x軸目盛を消す
    for i in range(len(axes2)):
        ax = axes2[i]
        row_data = df_rssi_t.iloc[i+1]
        Y = row_data.tolist()
        ax.bar(X, Y)  # 棒グラフ
        ax.set_yticks([70]) # y軸の目盛りを指定
        ax.set_ylim(50, 100)  # y軸の上限・下限を指定

    plt.subplots_adjust(wspace=0, hspace=0)  # 間を詰める
    plt.xlabel('time step')
    plt.ylabel('average RSSI at each BLE')

    #描画したグラフをpng形式で保存
    image_name = new_file_name.replace(".csv", "_rssi.png") # 要検討
    image_path = os.path.join(new_folder, image_name)
    plt.savefig(image_path, format="png", dpi=300)

    plt.close(fig2)

"""