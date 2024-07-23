# 経路推定の下準備（link_dataに対して）
# link dataに対して中点の座標を各linkに貼り付けて更新するためのコード
# x_o, x_d, x_midの各列を作る
import pandas as pd 

# 工事の前中後で変化するので注意
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv") # 偶数側と奇数側があるが同じ処理をした方が後で楽ではないか
df_node = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node_pre_1013.csv") # 位置修正済み

#### x_o列以降を削除した方がいい
# x_o_index = df_link.columns.get_loc('x_o')
# df_link = df_link.iloc[:, :x_o_index] # ilocは要素を数字で指定．locは要素を文字（index, column）で指定．x_o以降が削除
# df_link.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')

df_link = df_link.fillna(0)
# 各linkのO,Dの座標を取得
# 足して2で割ってmidの座標を取得
# linkにmidのx, y, z座標の値を貼り付ける

for i in range(len(df_link)):
    # ODを取得
    o = df_link.loc[i, 'O']
    d = df_link.loc[i, 'D']

    # ODの座標を取得
    # 3次元(z)の情報は一応つけておくがDDRの判定には用いない
    for j in range(len(df_node)):
        if df_node.loc[j, 'nodeid'] == o:
            x_o = df_node.loc[j, 'x']
            y_o = df_node.loc[j, 'y']
            z_o = df_node.loc[j, 'floor']

        if df_node.loc[j, 'nodeid'] == d:
            x_d = df_node.loc[j, 'x']
            y_d = df_node.loc[j, 'y']
            z_d = df_node.loc[j, 'floor']

    df_link.loc[i, 'x_o'] = x_o
    df_link.loc[i, 'y_o'] = y_o
    df_link.loc[i, 'z_o'] = z_o

    df_link.loc[i, 'x_d'] = x_d
    df_link.loc[i, 'y_d'] = y_d
    df_link.loc[i, 'z_d'] = z_d

    df_link.loc[i, 'x_mid'] =  (x_o + x_d) / 2
    df_link.loc[i, 'y_mid'] =  (y_o + y_d) / 2
    df_link.loc[i, 'z_mid'] =  (z_o + z_d) / 2

# print(df_link)
df_link.to_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv")