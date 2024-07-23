
##########################################
########## 観測結果からのODの集計 ###########
##########################################
import pandas as pd 
df = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/measuring_result.csv')
df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')

# dfのうち第2~51列目は不要
df = df.drop(df.columns[3:53], axis=1) 

# 滞在時間とo_link, d_linkを知りたいので各user_idの最後の列を入手したいー
grouped  = df.groupby('user_id')
dataframes_list = [group.reset_index(drop=True) for name, group in grouped] # ここで各dataframes_list内のindexをリセットする

print(dataframes_list[0])

# とりあえず第0行目を作ってみる→dfから1行適当に抜き出して後で消す
# row = df.iloc[0:4] # 行ごと抽出
row = df.iloc[[4], :]
# print(row.to_frame().T)
print(row)
"""
for each_data in dataframes_list:
    n = len(each_data)
    each_row = each_data.iloc[n-1, :] # 行ごと抽出
    # print(each_row)
    row = pd.concat([row, each_row], axis=1)
    #print(row) # concatは行と行の縦結合，listは処理できない

#print(row)

#row.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/tekitou.csv')
"""