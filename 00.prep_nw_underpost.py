# postとpreのリンクデータ生成
import pandas as pd 
import numpy as np 
"""
df_odds = pd.read_csv('shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
df_evens = pd.read_csv('shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')

# 工事中under
# ODいづれかに以下のノードが入っていたらそのリンクを削除する
delete_node_under = [12, 13, 14, 15, 16, 17, 30, 31, 35, 38]
df_odds_under = df_odds[~(df_odds['O'].isin(delete_node_under) | df_odds['D'].isin(delete_node_under))]
df_evens_under = df_evens[~(df_evens['O'].isin(delete_node_under) | df_evens['D'].isin(delete_node_under))]

# 工事後post
# ODいづれかに以下のノードが入っていたらそのリンクを削除する
delete_node_post = [12, 13, 14, 15, 16, 17, 19, 30, 31, 35, 38]
df_odds_post = df_odds[~(df_odds['O'].isin(delete_node_post) | df_odds['D'].isin(delete_node_post))] # ~で以降のブール値を反転させている．これがないと，delete_node_listに入ってる行
df_evens_post = df_evens[~(df_evens['O'].isin(delete_node_post) | df_evens['D'].isin(delete_node_post))] # ~で以降のブール値を反転させている．これがないと，delete_node_listに入ってる行

# この前に，link_evensを適切に変えておく．仮想リンクのところ．oddsには仮想リンク入ってないのでそのままで良さそうである．
# この機にlinkデータから座標の列を排除して観測モデル書き換えた方がいいのか？？？
# 毎回node情報を出させて参照させる方がいいのか？

df_odds_under.to_csv('shibuya_nw/shibuya_nw_detail/shibuya_link_under_odds.csv')
df_evens_under.to_csv('shibuya_nw/shibuya_nw_detail/shibuya_link_under_evens.csv')
df_odds_post.to_csv('shibuya_nw/shibuya_nw_detail/shibuya_link_post_odds.csv')
df_evens_post.to_csv('shibuya_nw/shibuya_nw_detail/shibuya_link_post_evens.csv')
"""

###### link_odds（観測モデル用）とlink_evens（oddsの反対方向+仮想リンク+吸収リンク）の合体 ######
post_odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_post_odds.csv')
post_evens = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_post_evens.csv')

# 二つのファイルをまずconcatして，link_id列の昇順で並べ替え，最後index振り直し
post_integrated = pd.concat([post_odds, post_evens]) # concatは合体させるdataframeをリストで与える
post_integrated = post_integrated.sort_values('linkid')

# 第0列目が不要なので削除
post_integrated = post_integrated.iloc[:, 1:] # 第1列目(link_id列)以降を保持
post_integrated = post_integrated.reset_index(drop = True)
post_integrated = post_integrated.fillna(0)
post_integrated.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_post_integrated.csv')

under_odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_under_odds.csv')
under_evens = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_under_evens.csv')

# 二つのファイルをまずconcatして，link_id列の昇順で並べ替え，最後index振り直し
under_integrated = pd.concat([under_odds, under_evens]) # concatは合体させるdataframeをリストで与える
under_integrated = under_integrated.sort_values('linkid')

# 第0列目が不要なので削除
under_integrated = under_integrated.iloc[:, 1:] # 第1列目(link_id列)以降を保持
under_integrated = under_integrated.reset_index(drop = True)
under_integrated = under_integrated.fillna(0)
under_integrated.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_under_integrated.csv')
