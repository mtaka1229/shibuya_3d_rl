###### nwデータの整理，特にz座標の付与 ######
###### 工事前に関してはすでにlink dataに座標適当に入れてしまってたので値を差し替える必要がある
import pandas as pd 
df_ble = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/ble_nw.csv')
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node_pre_1013.csv')
df_link_odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
df_link_evens = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')
# integratedはどうせ後で作り直しなので
# evensも後でやり直すが


# link_dataのwidthについて，列をまとめる．つまりwidth_staire*staire(dum)+width_home*
# これは効用関数で「階段の幅」「ホームの幅」という感じで分けた方がいいか．なのでこのままの方がむしろいいか．
# z座標に関しては，z_o, z_d, z_midを同様に変える
for i in range(len(df_link_evens)):
    if df_link_evens.loc[i, 'z_o'] == 1:
        df_link_evens.loc[i, 'z_o'] = 14.8
    if df_link_evens.loc[i, 'z_o'] == 2:
        df_link_evens.loc[i, 'z_o'] = 20.5
    if df_link_evens.loc[i, 'z_o'] == 3:
        df_link_evens.loc[i, 'z_o'] = 28.5

for i in range(len(df_link_evens)):
    if df_link_evens.loc[i, 'z_d'] == 1:
        df_link_evens.loc[i, 'z_d'] = 14.8
    if df_link_evens.loc[i, 'z_d'] == 2:
        df_link_evens.loc[i, 'z_d'] = 20.5
    if df_link_evens.loc[i, 'z_d'] == 3:
        df_link_evens.loc[i, 'z_d'] = 28.5

for i in range(len(df_link_evens)):
    if df_link_evens.loc[i, 'z_mid'] == 1:
        df_link_evens.loc[i, 'z_mid'] = 14.8
    if df_link_evens.loc[i, 'z_mid'] == 2:
        df_link_evens.loc[i, 'z_mid'] = 20.5
    if df_link_evens.loc[i, 'z_mid'] == 3:
        df_link_evens.loc[i, 'z_mid'] = 28.5
    if df_link_evens.loc[i, 'z_mid'] == 1.5:
        df_link_evens.loc[i, 'z_mid'] = 17.65
    if df_link_evens.loc[i, 'z_mid'] == 2.5:
        df_link_evens.loc[i, 'z_mid'] = 24.5

# oddsに対しても同じ処理
for i in range(len(df_link_odds)):
    if df_link_odds.loc[i, 'z_o'] == 1:
        df_link_odds.loc[i, 'z_o'] = 14.8
    if df_link_odds.loc[i, 'z_o'] == 2:
        df_link_odds.loc[i, 'z_o'] = 20.5
    if df_link_odds.loc[i, 'z_o'] == 3:
        df_link_odds.loc[i, 'z_o'] = 28.5

for i in range(len(df_link_odds)):
    if df_link_odds.loc[i, 'z_d'] == 1:
        df_link_odds.loc[i, 'z_d'] = 14.8
    if df_link_odds.loc[i, 'z_d'] == 2:
        df_link_odds.loc[i, 'z_d'] = 20.5
    if df_link_odds.loc[i, 'z_d'] == 3:
        df_link_odds.loc[i, 'z_d'] = 28.5

for i in range(len(df_link_odds)):
    if df_link_odds.loc[i, 'z_mid'] == 1:
        df_link_odds.loc[i, 'z_mid'] = 14.8
    if df_link_odds.loc[i, 'z_mid'] == 2:
        df_link_odds.loc[i, 'z_mid'] = 20.5
    if df_link_odds.loc[i, 'z_mid'] == 3:
        df_link_odds.loc[i, 'z_mid'] = 28.5
    if df_link_odds.loc[i, 'z_mid'] == 1.5:
        df_link_odds.loc[i, 'z_mid'] = 17.65
    if df_link_odds.loc[i, 'z_mid'] == 2.5:
        df_link_odds.loc[i, 'z_mid'] = 24.5


        
# df_nodeとdf_bleに対しては，'floor'列に対して，1→14.8, 2→20.5, 3→28.1に変換
for i in range(len(df_node)):
    if df_node.loc[i, 'floor'] == 1:
        df_node.loc[i, 'floor'] = 14.8
    if df_node.loc[i, 'floor'] == 2:
        df_node.loc[i, 'floor'] = 20.5
    if df_node.loc[i, 'floor'] == 3:
        df_node.loc[i, 'floor'] = 28.5

# ble
for i in range(len(df_ble)):
    if df_ble.loc[i, 'floor'] == 1:
        df_ble.loc[i, 'floor'] = 14.8
    if df_ble.loc[i, 'floor'] == 2:
        df_ble.loc[i, 'floor'] = 20.5
    if df_ble.loc[i, 'floor'] == 3:
        df_ble.loc[i, 'floor'] = 28.5

# 確認
print(df_link_odds)
print(df_link_evens)
print(df_node)
print(df_ble)
# 最後にcsvを更新
df_node.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node_pre.csv')
df_ble.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/ble_nw.csv')
df_link_odds.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
df_link_evens.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')
