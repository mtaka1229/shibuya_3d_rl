import pandas as pd 
import matplotlib.pyplot as plt 
import datetime

####################################################################
########## 30秒おきのRSSIの取れ方とユニークMACアドレス数の可視化 ###########
####################################################################
"""
df = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_raw/blescan_002_20221218.csv')

# x時台の全データを取り出す
df_17 = df[df['hour']==8]

df_17['TIME'] = pd.to_datetime(df_17['time'])

print(df_17)

start_time = '2022-12-18 8:00:00'
end_time = '2022-12-18 8:30:00'

df_limited = df_17[df_17['TIME'] <= end_time]

print(df_limited)
# [(df_8['TIME'] >= start_time) and (df_8['TIME'] <= end_time)]
# print(df_limited)
# print(df_limited.groupby(pd.Grouper(key='TIME', freq='1min')).count().reset_index())

result = df_limited.groupby(pd.Grouper(key='TIME', freq='30S')).agg({'RSSI': 'mean', 'MAC': 'nunique'}).reset_index()

result.columns = ['TIME', 'Average_RSSI', 'Unique_MAC_Count']

print(result)


result_df = result.reset_index()

# CSV ファイルに保存
result_df.to_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_08/dynamic_visualization2.csv', index=False)  # index 列を含めない場合は index=False を指定


#x = df_limited['TIME']
#print(x)
#y = df_limited['RSSI']

#plt.xlabel("TIME")
#plt.ylabel("RSSI")
# plt.xticks(range(, , ))
#plt.xticks(pd.date_range(start=x.min(), end=x.max(), freq='5T'), rotation=90)
#plt.yticks(range(-95, -40, 5))

#plt.title("relationship between time and RSSI at beacon No.002, 2022/12/19 8:00-8:10")

#plt.show()
df = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_08/dynamic_visualization2.csv')

#x = df['TIME']
#y1 = df['Average_RSSI']
#y2 = df['Unique_MAC_Count']

#plt.xlabel('TIME')
#plt.y1label("RSSI")

fig = plt.figure()
ax1 = fig.add_subplot()
t = df['TIME']
y1 = df['Average_RSSI']
ln1=ax1.plot(t, y1,'C0',label=r'Average_RSSI')

ax2 = ax1.twinx()
y2 = df['Unique_MAC_Count']
ln2=ax2.plot(t,y2,'C1',label=r'Unique_MAC_Count')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower right')

# x軸のメモリラベルを90deg回転
plt.xticks(rotation=90)
plt.title("at beacon no.2, 2022/12/18 08:00-08:30")


ax1.set_xlabel('t')
ax1.set_ylabel(r'Average_RSSI')
ax1.grid(True)
ax2.set_ylabel(r'Unique_MAC_Count')

#fig.show()

plt.show()
"""

###################################################################
########## 観測結果からのODの集計（result_analysis.csvの作成 ###########
###################################################################
"""
df = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/measuring_result.csv')
df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')

# dfのうち第2~51列目は不要
df = df.drop(df.columns[3:53], axis=1) # axis=1は列方向の処理を表す
# print(df)

# ユニークなユーザidを入手
# user_list = df['user_id'].unique().tolist()
# unique_user = len(user_list) 
# print(unique_user)：916

# home_link，gate_link
# home_link  = df_link[df_link['home'] == 1]
# gate_link = df_link[(df_link['chuo_gate'] == 1) |  (df_link['hachiko_gate'] == 1) | (df_link['minami_gate'] == 1)] # これ覚える！

# 滞在時間とo_link, d_linkを知りたいので各user_idの最後の列を入手したい．あるいは各user_idの先頭列の一個前の行を入手したい
grouped  = df.groupby('user_id')
dataframes_list = [group.reset_index(drop=True) for name, group in grouped] # ここで各dataframes_list内のindexをリセットする

# print(dataframes_list[1]) # ここまではちゃんとできている

# とりあえず第0行目を作ってみる→dfから1行適当に抜き出して後で消す
row = df.iloc[[4]] # 行ごと抽出

for each_data in dataframes_list:
    n = len(each_data)
    each_row = each_data.iloc[[n-1], :] # 行ごと抽出
    # print(each_row)
    row = pd.concat([row, each_row])
    #print(row) # concatは行と行の縦結合，listは処理できない

row = row.iloc[1:, 1:]
row = row.reset_index(drop = True)
print(row)

row.to_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/result_analysis.csv')
"""

####################################################
########## result_analysis.csvを使った集計 ###########
####################################################
"""
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/result_analysis.csv')
df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')

# home_link，gate_link
home_link  = df_link[df_link['home'] == 1]
gate_link = df_link[(df_link['chuo_gate'] == 1) |  (df_link['hachiko_gate'] == 1) | (df_link['minami_gate'] == 1)] # |をつかった書き方覚える！
hachi_link = df_link[df_link['hachiko_gate'] == 1]
chuo_link = df_link[df_link['chuo_gate'] == 1]
minami_link = df_link[df_link['minami_gate'] == 1]

# ODはhome_linkかgate_linkのどちらかのはず（観測モデルでstair_linkがOorDのデータは排除しているので
# 
"""
"""
# timestepの可視化
t = df['t']
# print(min(t))
#for tt in t:
t_counts = df['t'].value_counts().sort_index()

# plt.figure(figsize=(10, 6))  # グラフのサイズを設定 (オプション)
t_counts.plot(kind='bar')
plt.xlabel('time step')  # x軸ラベル
plt.ylabel('Count')  # y軸ラベル
plt.title('time step counts')  # グラフのタイトル

plt.show()
"""

"""
# O，Dの可視化（単純な推計）
t = df['d_link']
# print(min(t))
#for tt in t:
t_counts = df['d_link'].value_counts().sort_index()

# plt.figure(figsize=(10, 6))  # グラフのサイズを設定 (オプション)
t_counts.plot(kind='bar')
plt.xlabel('d')  # x軸ラベル
plt.ylabel('Count')  # y軸ラベル
plt.title('d counts')  # グラフのタイトル

plt.show()
"""
"""
# クロス集計
# oがhomeの人のd
df_oisgate = df[df['o_link'].isin(gate_link['link_id'])] # これあってる？
d_counts = df_oisgate['d_link'].value_counts().sort_index()

#print(d_counts)
d_counts.plot(kind='bar')
plt.xlabel('d')  # x軸ラベル
plt.ylabel('Count')  # y軸ラベル
plt.title('d counts of those whose o_is_gate')  # グラフのタイトル

plt.show()
# oがgateの人のd
"""

###################################################################
########## 混雑度densityの計算とデータ付与，および不要列カット ###########
###################################################################

df = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_1715.csv')
"""
df_parents = df_parents.iloc[:, 2:]

df_parents.drop(['num', 'type', 'day', 'hour', 'day_flag'], axis=1, inplace=True)
#df_parents = df_parents[df_parents[:, 2:]]
# 20221218_17（時間帯別の親データ）から，ビーコンごとの1分おきの混雑度＝ユニークmacアドレス数をカウント
# 
df_parents['ID'] = df_parents['ID'].str.replace('id', '').astype(int)
df_parents['time'] = pd.to_datetime(df_parents['time'])

df_parents.to_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_17.csv')
print(df_parents)

# 17:00-17:15のデータを取り出す．あまりに重いので．
df['time'] = pd.to_datetime(df['time'])
# cutline = datetime.datetime(2022, 12, 18, 17, 0, 0)
# Timestampはdatetimeをpandas上でも機能させるように作られたもの
cutline = pd.Timestamp('2022-12-18 17:15:00+09:00')

#print(df.loc[1, 'time'].dtype)
df_quater = df[df['time'] < cutline]
df_quater.to_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/20221218_1715.csv')
print(df_quater)
#print(df.dtypes)
"""
# dfは17:00-17:15までのデータ．綺麗に1/4になったが，それでもレコード数は100万を超えている．．
# まずはこのデータに対して10sec間隔でtimestepを与える（timestep列）

# time_step（10秒間隔）
# df['time'] = pd.to_datetime(df['time']) # datetime形式に変換
time_delta = 10                         
start_time = df.loc[0, 'time']
for i in range(len(df)):
    time_diff = (df.loc[i, 'TIME'] - start_time).total_seconds()
    df.loc[i, 'time_step'] = time_diff // (time_delta) +1
        

# alladdressも作り直しか？？
# この時点でtimestampが2以下のものは消してよさそう

