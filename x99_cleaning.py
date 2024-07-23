##########################################################################################
############もう用済み．検出回数しかないのでちょっと使いずらい．時系列とRSSIが重要#########################
##########################################################################################

#石井さんにいただいたmacアドレス毎に集計済みのデータ(blescan_macagg)をさらにクリーニングし余計なアドレスを削ぎ落とすコード

import pandas as pd
import os

#cleanedにはホームのみ検出のアドレスを消したデータがある．cleaned2は，改札階のみのやつも追加で削除したデータ
search_folder = "/Users/takahiromatsunaga/Desktop/shibuya/blescan_macagg_cleaned"
new_folder = "/Users/takahiromatsunaga/Desktop/shibuya/blescan_macagg_cleaned2"

#工事の前中後で観測機器の数が異なるので，IDのところは時期に応じて変更．ファイル名も変更
file_name = "/Users/takahiromatsunaga/Desktop/shibuya/blescan_macagg_cleaned/blescan_macagg_20230108_cleaned.csv"
df = pd.read_csv(file_name)

#IDのところは時期に応じて変更
new_df = df[(df.loc[:, 'ID_001':'ID_006'].sum(axis=1)  + df.loc[:, 'ID_022':'ID_023'].sum(axis=1)!= 0)]
new_name = file_name.replace("_cleaned.csv", "_cleaned2.csv")
new_file_path = os.path.join(new_folder, new_name)

new_df.to_csv(new_file_path, index=False)


"""""
#以下cleanedフォルダを作成したときのコード
#cleanedフォルダは本当に不要なので削除した(20230705)
for file_name in os.listdir(search_folder):
    file_path = os.path.join(search_folder,file_name)
    df = pd.read_csv(file_path)

    #indices_to_drop = []
    count=1のやつを消すと同時にホームでのみ検出されたaddressを削除する
    ホームのみで検出ということはつまり1For3Fコンコースのどちらでも観測されなかったということである
    より1F3Fでの検出回数がともに0であるデータは削除する 
  
    new_df = df[(df['count'] != 1) & (df['ID_007'] + df.loc[:, 'ID_012':'ID_021'].sum(axis=1) != 0)]
    new_name = file_name.replace(".csv", "_cleaned.csv")
    new_file_path = os.path.join(new_folder, new_name)

    new_df.to_csv(new_file_path, index=False)

    for i in range(len(df)):
        if (df.loc[i, 'count'] == 1) or (df.loc[i, "ID007"]+ df.loc[i, "ID012":"ID021"].sum() == 0):
            indices_to_drop.append(i)
    
    df = df.drop(indices_to_drop, axis=0)

    new_name = file_name.replace(".csv", "_cleaned.csv")
    new_file_path = os.path.join(new_folder, new_name)
    df.to_csv(new_file_path)
"""""

