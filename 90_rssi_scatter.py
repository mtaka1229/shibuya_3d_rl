
import pandas as pd 
import matplotlib.pyplot as plt 

#""""
# まず一旦特定のmacアドレスで分析
###############ここ要変更
df = pd.read_csv("/Users/takahiromatsunaga/bledata/blescan_20221219_18_cleaned/ e65d0c4cd6be99ed65821340c1ea23955f99e69ae3051e4a15b2c4387290dd7a.csv")

# time列のデータをdatetime型に変換してTIME列に格納
df['TIME'] = pd.to_datetime(df['time'])

# ID列からidをとって数字にする「id_num」列
df['id_num'] = df['ID'].str.replace('id', '').astype(int)

# id_numとRSSIの関係を可視化．in_numをTIMEにすると，時間とRSSIのグラフが描ける
X = df['id_num']
Y = df['RSSI']

plt.xlabel("id_num")
plt.ylabel("RSSI")
plt.xticks(range(2, 25, 2))

plt.scatter(X, Y)

plt.show()

#"""

