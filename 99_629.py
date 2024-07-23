########## 検出されている時間長をアドレスごとに割り出し，可視化 ###########

import pandas as pd 
import os
import datetime
import matplotlib.pyplot as plt

search_folder = "/Users/takahiromatsunaga/bledata/blescan_20221219_08_cleaned"

X = []
Y = []

for i, file_name in enumerate(os.listdir(search_folder)):
    file_path = os.path.join(search_folder, file_name)
    df = pd.read_csv(file_path)

    #time列のデータをdatetime型に変換してTIME列に格納
    df['TIME'] = pd.to_datetime(df['time'])

    #TIME列の昇順によって，全データを並べ替える
    df = df.sort_values('TIME', ascending=True).reset_index(drop=True)

    #ここでスタートタイムを切り捨て，エンドタイムを切り上げたい，その上で1分おきに集計したい
    start_time = df.loc[0, 'TIME'].replace(second=0, microsecond=0)
    end_time = df.loc[df.index[-1], "TIME"].replace(second=0, microsecond=0)
    #end_time = end_time + datetime.timedelta(minutes=1)

    X.append(i)
    Y.append(end_time - start_time)



plt.scatter(X, Y)
plt.xlabel('id of mac address')
plt.ylabel('time interval')
plt.title('the difference of time interval for each address')

plt.show()

