import pandas as pd 
import networkx as nx
import os
import datetime
import numpy as np
from datetime import timedelta 
import csv

"""
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import datetime

df = pd.read_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/alladdress/221.csv")
new_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/graph/221"

X = df['id_num']
Y = df['RSSI']

plt.xlabel("id_num")
plt.ylabel("RSSI")
plt.xticks(range(2, 25, 2))

plt.scatter(X, Y)

#plt.show()

rssi_scatter_name = "rssi_scatter.png"
new_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221219_08/graph/221"
rssi_scatter_path = os.path.join(new_folder, rssi_scatter_name)

# 保存
plt.savefig(rssi_scatter_path, format="png", dpi=300)
plt.close()
"""


##################################################################
################ 03. BLEidと検出回数との関係のグラフ #################
##################################################################
"""
# ここでもデータはすでに読み込んであるものを使うのでdfを継続して使用できる
# データフレームの設定
columns = ["id001", "id002", "id003", "id004", "id005", "id006", "id008", "id009", "id010", "id011", "id022", "id023", "id012", "id013", "id014", "id015", "id016", "id017", "id018", "id007", "id019", "id020", "id021"]

#countしたデータを入れるための空のデータフレームcsv
df_new = pd.DataFrame(columns=columns)

#平均rssiを出すための空のデータフレームcsv
#df_rssi = pd.DataFrame(columns=columns)

#ここでスタートタイムを切り捨て，エンドタイムを切り上げたい，その上で1分おきに集計したい
#start_time = df.loc[0, 'TIME'].replace(second=0, microsecond=0)
start_time = df.loc[0, 'TIME'].replace(df.loc[0, 'TIME'].split(':')[2], '00').split('.')[0]

#end_time = df.loc[df.index[-1], "TIME"].replace(second=0, microsecond=0)
end_time = df.loc[0, 'TIME'].replace(df.loc[df.index[-1], "TIME"].split(':')[2], '00').split('.')[0]

end_time = end_time + datetime.timedelta(minutes=1)

# 1分ごとのタイムステップ
delta = datetime.timedelta(minutes=1)

# 最初の時間だけはdf_newに行を追加しておく
df_new.loc[start_time] = 0
#df_rssi.loc[start_time] = 0

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
df_new_name = "count.csv"
df_new_path = os.path.join(new_folder, df_new_name)

# csvに出力
df_new.to_csv(df_new_path)

"""
######################################続いてrssiのファイルを作る############################################################
"""
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


"""
# 出力したdf_newを可視化 
df_new_t = df_new.T
X2 = [i+1 for i in range(len(df_new_t.columns))]

#（棒グラフが生える）横軸の本数を指定．これはidの数
fig, axes = plt.subplots(len(df_new_t)-1) #-1しないとラベルの列の分だけ列数が追加されてしまう

for ax in axes[:-1]:
    ax.tick_params(bottom=False, labelbottom=False)  # 一番下のax以外，x軸目盛を消す
for i in range(len(axes)):
    ax = axes[i]
    row_data = df_new_t.iloc[i+1]
    Y2 = row_data.tolist()
    ax.bar(X2, Y2)  # 棒グラフ
    ax.set_yticks([10]) # y軸の目盛りを指定
    ax.set_ylim(0, 20)  # y軸の上限・下限を指定

plt.subplots_adjust(wspace=0, hspace=0)  # 間を詰める
plt.xlabel('time step')
plt.ylabel('count at each BLE')

#描画したグラフをpng形式で保存
image_name = "count.png"
image_path = os.path.join(new_folder, image_name)
plt.savefig(image_path, format="png", dpi=300)

plt.close()

"""
##################################################
########## リンク接続行列作成関数の動作確認 ###########
##################################################
"""
import pandas as pd 
import networkx as nx
import os
import datetime
import numpy as np 

# a function generating connection matrix of the links
def link_connection(link_data): # link_data is a dataframe
    n = len(link_data)
    A = np.eye(n) # origin of connection matrix(対角成分が1)
    for i in range(n):
        O = link_data.loc[i, 'O']
        D = link_data.loc[i, 'D']
        # print(O, D)
        for j in range(n):
             # if (link_data.loc[j, 'O'] == O or D) or (link_data.loc[j, 'D'] == O or D): この書き方ダメらしい
            if ((link_data.loc[j, 'O'] == O) or (link_data.loc[j, 'O'] == D)) or (link_data.loc[j, 'D'] == O) or (link_data.loc[j, 'D'] == D):
                A[i, j] = 1
    return(A)

data = {
    'linkid' : [1, 2, 3, 4, 5, 6],
    'O' : [1, 2, 2, 3, 4, 4],
    'D' : [2, 3, 4, 6, 5, 6],
}

df_link = pd.DataFrame(data)

def link_candidate(link_data, linkid):
    #dict = {} # なんで辞書になってるんだ，リストでよくないか
    # indexs = []
    A = link_connection(link_data) # 用意するリンク接続行列
    b = np.zeros(len(link_data)) 
    b[linkid-1] += 1 
    candidate_bector = A.dot(b) # これは行ベクトルになっている．積はdot
    true_index = np.where(candidate_bector == 1)[0] + 1 # 行ベクトルの値が1の列のindex，linkidは1から始まるのでindex+1にするとtrue linkidになる．あとこれ勝手にリスト形式になってる
    #true_index = index + 1 # linkidは1から始まるのでindex+1にするとlinkidになる
    # indexs.append(true_index)
    #dict = {link_id: 0 for link_id in true_index}
    # link_dataのうち，'linkid'==list(list内にlinkidがある)の行を抽出する
    filtered_df = link_data[link_data['linkid'].isin(true_index)] # link_dataに'link_id'列があるかは不明だがなければerrorになるだけなのでこれでOKか？
    return(filtered_df) 

# print(link_candidate(df_link, 6)) # my_dict = {'A': 10, 'B': 30, 'C': 20, 'D': 5}
# ♪───Ｏ（≧∇≦）Ｏ────♪

# 値が最も大きい要素のキーを抽出
#max_key = max(my_dict, key=my_dict.get)
#print(f"値が最も大きい要素のキー: {max_key}", np.zeros(6))

df = pd.read_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/24.csv") 
grouped = df.groupby('ID')

# making the list of the dataframes
dataframes_list = [group for name, group in grouped] # これはちゃんと動く

# print(dataframes_list)  # OK

print(df['TIME'].dtype)

"""

"""
import numpy as np 
count = 100
dict = {'car':3, 'train':10, 'bus':5}
for key in dict:
        dict[key] = dict[key]/count
#print(dict)

array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 各列の和を計算
column_sums = np.sum(array, axis=0)

# print(column_sums)
time_step = 10
time_step_array = np.array([time_step])

# time_stepをcolumn_sumsの先頭に追加
result = np.concatenate((time_step_array, column_sums))
res1 = [3, 5, 6, 7]
numpy_tot = np.vstack((result, res1))

print(numpy_tot)
"""

"""
# a function generating link connection matrix from link data
def link_connection(link_data):
    n = len(link_data)
    A = np.eye(n)
    for i in range(n):
        O = link_data.loc[i, 'O']
        D = link_data.loc[i, 'D']
        for j in range(n):
            if ((link_data.loc[j, 'O'] == O) or (link_data.loc[j, 'O'] == D)) or (link_data.loc[j, 'D'] == O) or (link_data.loc[j, 'D'] == D):
                A[i, j] = 1
    return(A)

# a function generating the link_id list of candidates
def link_candidate(link_data, linkid):
    A = link_connection(link_data) # 用意するリンク接続行列
    b = np.zeros(len(link_data))
    b[linkid-1] += 1
    candidate_bector = A.dot(b) # 積はdot
    true_index = np.where(candidate_bector == 1)[0] + 1 # linkidは1から始まるのでindex+1にする．リスト形式
    filtered_df = link_data[link_data['linkid'].isin(true_index)]
    filtered_df = filtered_df.reset_index() # indexリセットする

    return(filtered_df)
"""
#######################################
########## 観測モデルの動作確認 ###########
#######################################
"""
# a function culculating DDR from RSSI
# 暫定．ひとまずこれで推定してみる
def ddr(rssi):
    return (3*rssi+110)/(-1*8)

# reading NW data
df_node = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node_pre2.csv")
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv")

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_08/1alladdress_test"
file_list = list(os.listdir(search_folder))

output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_08/route_probability_test3"
# 一応，userごとにファイル
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# route estimation
for file_name in file_list:
    # 各userのパス（経路全体）の確定
    file_path = os.path.join(search_folder, file_name)

    # dataの読み込み
    df = pd.read_csv(file_path) 
    df['TIME'] = pd.to_datetime(df['TIME']) # datetime形式に変換

    # time_stepを幅10"で振り分ける．途中飛ばさない
    time_delta = 10

    n = len(df)

    start_time = df.loc[0, 'TIME']
    for i in range(len(df)):
        time_diff = (df.loc[i, 'TIME'] - start_time).total_seconds()
        df.loc[i, 'time_step'] = time_diff // (time_delta) +1 # これで10秒おきのtimestep（途中欠損部分も飛ばさない）が振られる

    # grouping by 'time_step'
    grouped = df.groupby('time_step') # groupbyメソッドで分割してもindexは保たれる

    # making the list of the dataframes of each timestep
    dataframes_list = [group.reset_index(drop=True) for name, group in grouped] # ここでindexをリセットする

    # くっつけるためのnumpy
    numpy_tot = np.zeros(len(df_link)+1) # timestep列が先頭に来るので長さを+1している

    for i in range(len(dataframes_list)):
        # 各time_stepにおけるリンクの観測確率の算定

        # timestep数
        num_timestep = len(dataframes_list)

        # timestep i の部分のデータ
        df_at_time = dataframes_list[i] 

        # time_stepスタンプはdf_at_timeで共通なので
        time_step = df_at_time.loc[0, 'time_step']
      
        # くっつけるためのnumpy
        numpy = np.zeros(len(df_link))

        for j in range(len(df_at_time)):
            # timestep内の個々の時点j/beacon xにおける近いリンクを捜索
            # RSSIを取得
            rssi = df_at_time.loc[j, 'RSSI']

            likelihood = np.zeros(len(df_link))

            # 各データiの検出bleの座標を取得
            x_ap = df_at_time.loc[j, 'x']
            y_ap = df_at_time.loc[j, 'y']
            z_ap = df_at_time.loc[j, 'floor']*1000 
            # zの値をめちゃ大きくすることで，階を挟んだ電波捕捉を捨象（階の厳密な高さが不明なのと，床板・天井版による電波遮蔽を考慮するため）．階が同じなら0になるので変な影響はない

            for k in range(len(df_link)):
                linkid = df_link.loc[k, 'link_id']
                # linkidは奇数なのでindexはlinkid//2+1
                index = linkid // 2 + 1

                x_o = df_link.loc[k, 'x_o']
                y_o = df_link.loc[k, 'y_o']
                z_o = df_link.loc[k, 'z_o']*1000

                x_d = df_link.loc[k, 'x_d']
                y_d = df_link.loc[k, 'y_d']
                z_d = df_link.loc[k, 'z_d']*1000

                x_mid = df_link.loc[k, 'x_mid']
                y_mid = df_link.loc[k, 'y_mid']
                z_mid = df_link.loc[k, 'z_mid']*1000
               
                # 1/4点
                x_qu = x_o/4 + x_d/4*3
                y_qu = y_o/4 + y_d/4*3
                z_qu = (z_o/4 + z_d/4*3) * 1000

                # 3/4点
                x_qua = x_o/4*3 + x_d/4
                y_qua = y_o/4*3 + y_d/4
                z_qua = (z_o/4*3 + z_d/4) * 1000

                # 距離を3種類計算→zの距離をめちゃ大きくすることで階が違ったら弾かれるようにする
                d_mid = ((x_mid - x_ap)**2 + (y_mid - y_ap)**2 + (z_mid - z_ap)**2)**0.5
                d_o = ((x_o - x_ap)**2 + (y_o - y_ap)**2 + (z_o - z_ap)**2)**0.5
                d_d = ((x_d - x_ap)**2 + (y_d - y_ap)**2 + (z_d - z_ap)**2)**0.5
                d_qu = ((x_qu - x_ap)**2 + (y_qu - y_ap)**2 + (z_qu - z_ap)**2)**0.5
                d_qua = ((x_qua - x_ap)**2 + (y_qua - y_ap)**2 + (z_qua - z_ap)**2)**0.5

                # 5種類のうち最小のものがddrより内側にあれば良い
                d = min(d_mid, d_o, d_d, d_qu, d_qua)

                if d < ddr(rssi):
                    likelihood[index-1] = 1

            sums = likelihood.sum()    # sumsが0だと割り算でエラーになるので回避
            if sums == 0:
                break
            likelihood = likelihood/sums

            # time_step内で尤度出すので，個々のデータで得られた結果をループが移行する前に足しておく
            numpy = np.vstack((numpy, likelihood))

            ## ここまでで各個別のログデータに対する処理が終了→次のログデータへ
            
        # timestep内の各尤度を合計する，つまりnumpyの各列の和を取る．numpy[i]がtime_step i における各linkの観測尤度の尤度の和になっている
        column_sums = np.sum(numpy, axis=0)

        # 要素数は1．numpynumpy配列化するために実行
        time_step_array = np.array([time_step])

        # time_stepをcolumn_sumsの先頭に追加
        result = np.concatenate((time_step_array, column_sums))

        # numpy_tot にtimestep i の尤度情報を付け足し
        numpy_tot = np.vstack((numpy_tot, result))
    
    # numpy_totの初期設定で入れておいた0の列を消去
    numpy_tot = np.delete(numpy_tot, 0, axis=0)
    # ここまででnumpy_totに各timestepの各リンクの観測尤度が出ている
    
    # csvファイル化を見据え，numpyをdataframe化
    df_result = pd.DataFrame(data=numpy_tot)

    # columnsを設定
    df_result.columns = ['t'] + [str(i) for i in range(1, 100) if i % 2 != 0]

    # 第0列目はtなので，link尤度の情報が入ってるのは第1~51列
    # o, dそれぞれtimestepの先頭・最後尾で尤度が最大であったものを取得
    # 尤度が最大のものが複数あることも考えられるが，その場合idxmaxメソッドは見つかった最初のindexを返す．
    # linkの設定から，階段は後ろの方に設定しておいたので，コンコースかホーム上のリンクが優先されるから，階段とホームの尤度が最大であったときもホームを採用するから問題なし
    # 1:51で1~51列目までを検索する
    o_link = df_result.iloc[0, 1:52].idxmax() 
    d_link = df_result.iloc[num_timestep-1, 1:52].idxmax()

    # o_linkあるいはd_linkが階段リンクだった場合，to_csvをしないでループを飛ばす
    # 階段リンク：idが71-99の奇数
    stairs = [i for i in range(70, 100) if i % 2 != 0]

    if (o_link in stairs) or (d_link in stairs):
        break

    # 最後にこのuserのo_linkとd_linkを入れるための列を追加
    df_result['o_link'] = o_link
    df_result['d_link'] = d_link

    result_path = os.path.join(output_folder, f"{file_name}")
    df_result.to_csv(result_path)

    print(f"{file_name}.csv に保存しました")

    # 以上であるuserに対する処理が終了
"""
########################################
########## selected_csvの作成 ###########
########################################
"""
import os
import shutil

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/alladdress"

# alladdressから上から1000個データ取り出してフォルダ作ってそこで回してみる
# testファイルでは回ったのにalladdressで回らないの普通に理解できなさすぎる

# ソースフォルダと新しいフォルダのパスを指定
output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/selected"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# ソースフォルダ内のファイルを取得
files = os.listdir(search_folder)

# 最初の1000個のCSVファイルをコピー
count = 0
for file in files:
    if file.endswith(".csv"):
        source_file = os.path.join(search_folder, file)
        destination_file = os.path.join(output_folder, file)
        shutil.copy(source_file, destination_file)
        count += 1
        if count == 1000:
            break

print(f"{count}個のCSVファイルを新しいフォルダに複製しました。")
"""
"""
import os
read_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/measuring_result"
file_list = list(os.listdir(read_folder))

print(len(file_list))
"""

####################################
########## リンクの偶奇合体 ###########
####################################
"""
import pandas as pd 

odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
evens = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')

# 二つのファイルをまずconcatして，link_id列の昇順で並べ替え，最後index振り直し
integrated = pd.concat([odds, evens]) # concatは合体させるdataframeをリストで与える
integrated = integrated.sort_values('link_id')
# print(integrated['link_id'].dtype) # link_idはint64型

# 第0, 1行目が不要なので削除
integrated = integrated.iloc[:, 2:] # 第2列目(link_id列)以降を保持

# やはりindexgが揃わないのでリセット
integrated = integrated.reset_index(drop = True)

#print(integrated)

integrated.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_integrated.csv')
"""

#####################################
########## 観測方程式の検討 ###########
#####################################
"""
import math
import matplotlib.pyplot as plt 

def rssi(D):
    return (-50-30*math.log10(D))

####### 観測方程式 #######
# これがDDR？観測rssiと周辺densから，各ビーコンからのおおよその距離を推定する関数
#def dist_from_rssi(rssi):
#    return (10**((-rssi-50)/30))


x = [i for i in range(1, 50)]
y = [rssi(d) for d in x]

plt.scatter(x, y)
plt.xlabel('distance')
plt.ylabel('RSSI')
plt.title('RSSI function')
plt.show()
"""
"""
read_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address"
file_list = list(os.listdir(read_folder))

for file_name in file_list:
    file_path = os.path.join(read_folder, file_name)
    df = pd.read_csv(file_path)

    df = df.iloc[:, 2:]

    df.to_csv(file_path)
"""
#########################################################
########## NWデータの階段リンクへのリンク長情報付与 ###########
#########################################################
"""
import pandas as pd 

df_odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
df_evens = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')

# ['length']列のうち，0の行を10にする
# 階段は長さ同じ（ほぼ）
for i in range(len(df_odds)):
    if df_odds.loc[i, 'length'] == 0:
        df_odds.loc[i, 'length'] = 10

for i in range(len(df_evens)):
    if df_evens.loc[i, 'length'] == 0:
        df_evens.loc[i, 'length'] = 10

print(df_evens)
print(df_odds)

df_odds.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
df_evens.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')

# OK-
"""

####### 効用関数の可視化してみた #######
"""
df_link_odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
X = [-1, 1, 1, 10]
def linkv(X, df_link_odds): # 効用関数は検討中
    vinstant = df_link_odds['length'] * X[0] + df_link_odds['width_home'] * X[1] + df_link_odds['width_staire'] * X[2] + df_link_odds['staire_with_esc'] * X[3]
    vinstant = np.exp(vinstant)
    return vinstant # 出力形式はpandasのseries

def Mset(X, df_link_odds):
    cost = linkv(X, df_link_odds)
    cost = pd.concat([cost]*50, axis=1) # axis=1は横方向
    cost = cost.T
    return cost

print(Mset(X, df_link_odds))
#test.to_csv("/Users/takahiromatsunaga/res2023/test/Mset_test.csv")
# なぜか列方向に足されている
#print(Mset(X, df_link_odds))

#LinkV = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
#print(LinkV)
# LinkV を横方向に10個並べて10x10の行列を作成
#LinkV_matrix = np.tile(LinkV, (1, 10))
#LinkV_dataframe = pd.DataFrame(LinkV_matrix)
# LinkV_matrix を表示
#print(LinkV_dataframe)
"""
"""
L=10 
D=2
V = np.zeros((L, D))
#z = np.exp(V)
z = np.ones((L, D)) 
print(z) 
"""

### np.sum(np.abs(zdd-zd))の動作確認
"""
zdd = np.array([1, 2, 3, 4, 5])
zd = np.array([10, 12, 1, 0, 2]) # np.arrayで配列作るやり方

dl = np.sum(np.abs(zdd-zd))

print(np.abs(zdd-zd))

print(dl)
"""

##### 価値関数や対数尤度関数のパーツの動作確認
"""
V = np.zeros((5, 2)) # exp(0)=1よりexp(-1)は小さくなるから?
# z = np.exp(V)

M = np.full((5, 5), 4)
M[:, 2] = np.exp(0)
# print(V[:, 1]) # これは列ベクトルと期待してるがここで行ベクトルになってる気がする
# やっぱりそうだ
z = np.exp(2**V[:, 1]).reshape(5, 1) 
#print(z)
#print(z.shape) # 5*1のはず

ZD = np.tile(z, (1, 10)) # これなら5*10のはず

print(M)
print(z) # 5*1のはず

Mz = (M @ z != 0) * (M @ z) + (M @ z == 0) # @に関してはzが行でも列でも問題ない
# これが5*1になってない？？？？→なってた，ちゃんと．
print(Mz)
print(Mz.shape)
#Mzt = Mz.T
#print(Mzt)

MZ = np.tile(Mz, (1, 10)) # 5*10だね

print(MZ)
print(MZ.shape) # 5*10

M2 = np.ones((5, 10))

p = (M2*ZD)/MZ

print(p.shape)
"""

# どうでもいい確認
"""
staire_link1 = [i for i in range(1, 100) if i % 2 != 0]
staire_link2 = [i for i in range(1, 100, 2)]

print(staire_link1)
print(staire_link2)

#[str(i) for i in range(1, 100) if i % 2 != 0]
"""

### 観測モデル，非連続の場合の処理の確認 ###

"""
prev_result = np.array([0, 0.5, 0.2, 0, 0, 0, 0.1, 0.1, 0, 0.05, 0.05, 0, 0]) # res_unionとったら実際こんな感じのはず

nonzero_indices = np.where(prev_result != 0)[0] # 1, 2, 6, 7, 9, 10のはず．# ここから観測確率!=0のリンクのindexを得られた

#print(nonzero_indices)

prev_link = nonzero_indices*2 + 1 # 元々のlinkid

#print(prev_link) # 3 5 13 15 19 21


index = prev_link//2


### 観測モデル，非連続の場合の処理の確認 ###
link_candidates = [3, 5, 9, 13, 17, 19, 21]      # linkid
# linkidからindexを得る
#cand_index = link_candidates//2     # 1, 2, 4, 6, 8, 9, 10
cand_index = [x//2 for x in link_candidates] # リストは直接割れない！
# 判定用．連続の場合の処理をまねる
judge = np.zeros(15)
print(judge)

index = np.array([i for i in range(15)]) # 全部のリンクのindex
mask = np.isin(index, cand_index)
judge[index[mask]] = 1

judge_sum = np.sum(judge) # 7
link_probability = judge/judge_sum 

print(link_probability) # 1, 2, 4, 6, 8, 9, 10に1/7=0.142辺りが入るはず


time_step_array = np.array([14])
t_result = np.concatenate((time_step_array, link_probability))
                    
user_id_array = np.array([2003])
t_result_withid = np.concatenate((user_id_array, t_result))


print(t_result_withid)

"""
###### link_odds（観測モデル用）とlink_evens（oddsの反対方向+仮想リンク+吸収リンク）の合体 ######
"""
odds = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv')
evens = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_evens.csv')

# 二つのファイルをまずconcatして，link_id列の昇順で並べ替え，最後index振り直し
integrated = pd.concat([odds, evens]) # concatは合体させるdataframeをリストで与える
integrated = integrated.sort_values('link_id')
# print(integrated['link_id'].dtype) # link_idはint64型

# 第0, 1, 2列目が不要なので削除
integrated = integrated.iloc[:, 3:] # 第3列目(link_id列)以降を保持

# やはりindexgが揃わないのでリセット
integrated = integrated.reset_index(drop = True)

integrated = integrated.fillna(0)

integrated = integrated.iloc[:, 1:] # 0列目不要
print(integrated)

integrated.to_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_integrated.csv')
"""
###### virtual connectの動作確認 ######
"""
virtual_connect = np.array([[6, 2, 1, 7],
                            [5, 6, 8, 8]
                            ])

d_link = 7  # d_linkと比較する値

# 最初の行の各要素がd_linkと等しいかどうかを判定し、True/Falseのブール配列を作成
is_equal = virtual_connect[0] == d_link

# Trueの要素の列番号を取得
equal_columns = np.where(is_equal)[0] # リスト？配列？


# d_linkに応じて対応する仮想リンクに確率1を与える
columns_index = equal_columns[0] # int(equal_columns[0]) # columns_indexのデータ形式は？→int
virtual_link = virtual_connect[1, columns_index] # これで対応する仮想リンクのlinkidが得られるはず．つまり5

print(columns_index)
print(virtual_link)

"""
# for i in range(9, -1, -1): # 要素数は9，9~1
    # print(i)

# for k in range(1, 100, 2):
    # print(k)

###### measuring model2の一部の動作確認 ###### 
"""
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node_pre.csv')
df_link_integrated = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_integrated.csv')

next_maxlink = 13

print(df_link_integrated['linkid'])
print(df_link_integrated['linkid'].shape)

maxlink_onode = df_link_integrated[df_link_integrated['linkid'] == next_maxlink]['O'].values # 13のonodeは8
# valuesをつけないとindexもついてくる．valuesをつけるとindexがつかない，ただしリストに入った形で返される
maxlink_onode = df_link_integrated[df_link_integrated['linkid'] == next_maxlink]['O'].iloc[0] # 13のonodeは8


print(maxlink_onode) # 8（ちゃんと出ている）→series型になってる気がする
print(maxlink_onode.dtype)

# onode_x = df_node[df_node['nodeid'] == maxlink_onode]['x'] # やはりこれだとindexもついてくる
onode_x = df_node[df_node['nodeid'] == maxlink_onode]['x'].iloc[0]
onode_y = df_node[df_node['nodeid'] == maxlink_onode]['y'].iloc[0]
onode_z = df_node[df_node['nodeid'] == maxlink_onode]['floor'].iloc[0]

print(onode_x) # -11917.391のはず
onode_loc = (onode_x, onode_y, onode_z)
            ## これOK???
print(onode_loc) # -11917.391,-37938.749,20.5 #### OKOKOKOKOKOKOKOKOKOk!
print(list(onode_loc))
zahyou = np.array(list(onode_loc))

print(zahyou)
"""


###### 1700-1715から100userを抽出 
"""
read_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address"
file_list = list(os.listdir(read_folder))
output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address200"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 最初の1000個のCSVファイルをコピー
count = 0
for file in file_list:
    if file.endswith(".csv"):
        source_file = os.path.join(read_folder, file)
        destination_file = os.path.join(output_folder, file)
        shutil.copy(source_file, destination_file)
        count += 1
        if count == 200:
            break

print(f"{count}個のCSVファイルを新しいフォルダに複製しました。")
"""

"""
import time 
start_time = time.time()
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
#print(df)
# DataFrameをNumPy配列に変換
numpy_array = df.values

end_time = time.time()

proc_time = end_time - start_time

print(numpy_array)
print(proc_time)

if isinstance(numpy_array, np.ndarray):
    print("arrはNumPy配列です")
"""
"""
import math
print(math.log(0.01))
"""
###### 1700-1715から100userを抽出 
"""
read_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address_gateend4"
file_list = list(os.listdir(read_folder))
output_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address_gateend500"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 最初の100個のCSVファイルをコピー
count = 0
for file in file_list:
    if file.endswith(".csv"):
        source_file = os.path.join(read_folder, file)
        destination_file = os.path.join(output_folder, file)
        shutil.copy(source_file, destination_file)
        count += 1
        if count == 500:
            break
        
print(f"{count}個のCSVファイルを新しいフォルダに複製しました。")


# address_gate_endの各csvから最初の変な行をとる



#read_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address_gateend4"
#file_list = list(os.listdir(read_folder))

#print(len(file_list))
"""

# データいくつ入ってるのか確認
import pandas as pd 
df = pd.read_csv('/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_gps/old/10028_20230128/t_loc_data.csv', encoding='shift-jis')
nuser = df['ユーザーID'].nunique()
print(nuser)