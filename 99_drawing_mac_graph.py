##########################################################################################
#######################こちらももう用済み．検出回数より時系列とRSSIが重要#########################
##########################################################################################
#macユーザごとの検出回数のグラフを描画するためのコード

import matplotlib.pyplot as plt
import pandas as pd 

df_new = pd.read_csv("/Users/takahiromatsunaga/bledata/test.csv")
df_new_t = df_new.T 
df_new_t.to_csv("/Users/takahiromatsunaga/bledata/tenchi.csv")

X = [i+1 for i in range(len(df_new_t.columns))]

#（棒グラフが生える）横軸の本数を指定．これはidの数
fig, axes = plt.subplots(len(df_new_t)-1)

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

plt.show()