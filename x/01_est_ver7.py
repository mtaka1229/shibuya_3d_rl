##### 同時推定コード #####
# ver1128．ついにちゃんとした尤度関数（交差エントロピー）の定式化ができそう．それに基づいた推定コード

import pandas as pd 
import os
import datetime
import numpy as np
from datetime import timedelta 
import csv
import time
import math
from scipy.stats import norm
from scipy.optimize import minimize
import random

start_time = time.time()
print('start!')

####### reading data #######
# inputも時間間隔20secで実行
df_quater = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20230130_08/20230130_0845_20sec.csv')
search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20230130_08/20230130_0845_20sec_end162021"
file_list = list(os.listdir(search_folder))

# df_linkも作り直し
df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_alllink_post.csv')
L = len(df_link)

# df_nodeは作り直し
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_node.csv')

df_ble = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/ble_nw.csv')

matrix = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_matrix_post.csv')

###### 各linkのoとdの座標を入れておく配列 ######
oddslink_loc_array = [] 
for i in range(len(df_link)):
    linkid = df_link.loc[i, 'linkid']
    O = df_link.loc[i, 'o']
    D = df_link.loc[i, 'd']

    x_o = df_node[df_node['nodeid'] == O]['x'].iloc[0]
    y_o = df_node[df_node['nodeid'] == O]['y'].iloc[0]
    z_o = df_node[df_node['nodeid'] == O]['floor'].iloc[0] 
    o_loc = np.array([x_o, y_o, z_o])

    x_d = df_node[df_node['nodeid'] == D]['x'].iloc[0]
    y_d = df_node[df_node['nodeid'] == D]['y'].iloc[0]
    z_d = df_node[df_node['nodeid'] == D]['floor'].iloc[0] # これでseriesではなく値として抽出できるはず
    d_loc = np.array([x_d, y_d, z_d])

    loc_tuple = (o_loc, d_loc)

    oddslink_loc_array.append(loc_tuple) # oddslink_loc_array[linkid//2][0]がlinkidのonode, [1]がdnode，[2]が中点，，，，

###### ビーコンとリンク線分の中点間距離を返す関数 ######
def shortest_distance_to_segment(p1, p2, x):
    p1 = np.array(p1)
    p2 = np.array(p2)
    x = np.array(x)
    
    segment = p2 - p1
    v1 = x - p1
    v2 = x - p2
    mid_point = (p1 + p2) / 2

    # 垂直ベクトルを計算
    v = v1 - np.dot(v1, segment) / np.dot(segment, segment) * segment
    
    distance = np.linalg.norm(x - mid_point)
    
    return distance

###### d_arrayの用意 ###### # 各リンク(の中点）とbleビーコンとの距離配列
d_array = np.zeros((len(df_link), len(df_ble)))

for i in range(len(df_link)):
    p_o = oddslink_loc_array[i][0] # o座標 # i=1の時
    p_d = oddslink_loc_array[i][1] # d座標

    for j in range(len(df_ble)):
        x_ap = df_ble.loc[j, 'x']
        y_ap = df_ble.loc[j, 'y']
        z_ap = df_ble.loc[j, 'floor']
        p_ap = np.array([x_ap, y_ap, z_ap])

        d_array[i, j] = shortest_distance_to_segment(p_o, p_d, p_ap) # ijはlink i+1, ビーコンj+1の距離

###### 混雑度dens，maxRSSI，meanRSSI，検出回数を各ビーコン各timestepごとに入れた配列（ビーコン数23*タイムステップ数） ######
### 20秒にするのでここやり直す．
dens_jt = np.zeros((23, 45), dtype = int)
max_jt = np.zeros((23, 45), dtype = int)
mean_jt = np.zeros((23, 45), dtype = int)
#count_jt = np.zeros((23, 45), dtype = int)

grouped = df_quater.groupby(['ID', 'timestep']) # listにして渡す必要がある
df_list = [group.reset_index(drop=True) for name, group in grouped]

for group in df_list:
    # beacon_idとtimestepを得る
    beacon_id = int(group.loc[0, 'ID'])
    timestep = int(group.loc[0, 'timestep'])

    # ユニークMACアドレス数，最大&平均RSSI
    unique_mac_count = group['MAC'].nunique()
    max_rssi = group['RSSI'].max()
    mean_rssi = group['RSSI'].mean()
    # countはそのタイムステップでの全検出数つまりgroupの長さそのもの
    #count = len(group)

    # numpy配列の要素はシーケンスダ
    dens_jt[beacon_id-1, timestep-1] = unique_mac_count
    max_jt[beacon_id-1, timestep-1] = max_rssi
    mean_jt[beacon_id-1, timestep-1] = mean_rssi
    #count_jt[beacon_id-1, timestep-1] = count

###### 接続行列 ###### この前にmatrix dataを作成する必要がある．
#I = np.eye(L)  # LxLの0で初期化された行列を作成．滞在をOKにするので対角成分1
#for i in range(matrix.shape[0]):
#    kn = matrix[i, 0]
#    an = matrix[i, 1]
#    k = np.where(linkid == kn)[0]  # knに対応するインデックスを取得
#    a = np.where(linkid == an)[0]  # anに対応するインデックスを取得
#    I[k, a] = 1  # 対応する位置に1を設定

####### リンク接続行列 ####### 
def Imake(link_data):
    n = len(link_data)
    I = np.eye(n)
    for i in range(n):
        D = link_data.loc[i, 'd']
        for j in range(n):
            if link_data.loc[j, 'o'] == D: 
                I[i, j] = 1
    return(I)

###### 接続行列 ###### 吸収リンクから吸収リンクへの接続は0に直す
I = Imake(df_link) 

# 最後に吸収リンク同士および階段リンク同士の接続は変なので最後のところ0にする
for i in range(51, 100):
    I[i-1, i-1] = 0

###### 改札の吸収リンク
ddata = [97, 98, 99]  

D = len(ddata)

####### パラメタ初期設定 #######
xdim = 2
x0 = np.zeros(xdim)

####### 観測方程式 ###### （観測確率．リンクkにいるとき，最大スコアのビーコンがjになる確率）
def mq(dist, distmin):     # 基本的にdistに反比例するか，それか何かの分布を与える．とりあえず試すしかないわ．
    return (distmin / dist) # これはリンクとビーコンの距離d_arrayを読み込めばOKかな

####### 観測方程式（ビーコンの評価） ######
def beacon_eval(rssi, max_rssi, freq): # 同時刻内で捕捉されたビーコンを評価．最大rssiと観測頻度による．20secも幅があるので
    return (max_rssi/rssi + freq) 

####### 観測モデル ###### # 各個人の各tiemstepに対して最もスコアの高いビーコンを決めるので出力はuserid, timestep, maxbeacon, absorption
def measuring_model(x):
    print(f'paramaterは{x}')
    qall = np.zeros(4)
    for file_name in file_list:
        if file_name.endswith('.csv'):
            base_name = file_name.split('.')[0]     # 数字部分のみ
            userid = int(base_name) # 後で1列目に入れる
                    
            file_path = os.path.join(search_folder, file_name)
            dfi = pd.read_csv(file_path) 
            nts = dfi['timestep'].nunique()                

            grouped = dfi.groupby('timestep')
            dfi_list = [group.reset_index(drop=True) for name, group in grouped]  
            qi = np.zeros(4)  # userid, timestep, maxbeacon, absorptionを後で入れる

            for t in range(nts):
                dfit = dfi_list[t]
                ts = int(dfit.loc[0, 'timestep']) # これintになってるか？は要確認

                count_tot = len(dfit) # 個人i時点tのcountの総和はdfitの長さそのもの
                        
                result_beacon = dfit.groupby('ID')
                result_beacon_list = [group.reset_index(drop=True) for name, group in result_beacon]

                ### 確認
                # print(result_beacon_list)

                # 各ビーコンのscoreを記録するためのdict
                dict = {}

                for j in range(len(result_beacon_list)): # 各ビーコンの結果を束ねたdataframe
                    dfitj = result_beacon_list[j]
                    count = len(dfitj) # そのビーコンでの検出回数
                    jid = dfitj.loc[0, 'ID']
                    rssi = dfitj['RSSI'].max() # そのビーコンでの最大観測強度

                    max_rssi = max_jt[jid-1, ts-1] # tsでのbeaconのmaxのrssi
                    freq = count / count_tot

                    # このビーコンのスコアを計算
                    jscore = beacon_eval(rssi, max_rssi, freq)

                    # dictにjidとjscoreを入れる
                    # dict.update(jid = jscore)
                    dict[jid] = jscore

                # dictに全部入っているので比較
                max_beacon = max(dict, key = dict.get) # score maxのビーコン

                # 個人i時刻tの結果（最もスコアの高かったビーコン）．absorptionの列はとりあえず0のまま残しておく
                qit = np.array([userid, ts, max_beacon, 0])
                qi = np.vstack((qi, qit))
            
            # qiの冒頭の0の行を削除
            qi = np.delete(qi, 0, axis = 0)

            # ここまでで終了したので，qiの最後の行のmaxbeaconの値を取得しそれを最後の列に入れる
            abs = qi[-1, 2] # これで最後の行のmaxbeaconの列（0から数えて2列目）
            # qiの最後の列の値を全てabsにする
            qi[:, -1] = abs

            # 全個人に対してvstackしていってqallを作る
            qall = np.vstack((qall, qi))

    # 冒頭の0の行を削除
    qall = np.delete(qall, 0, axis = 0)
    return qall 

###### 経路選択モデル ####### 
###### 初期設定 ######
V0 = np.full((L, D), -1) # exp(0)=1よりexp(-1)は小さくなるから?
z0 = np.exp(V0)

###### TSを設定しないといけない ###### #### 一斉に発生すると考えて良さそう→とりあえず時間構造化しないで解く
#TS = 10

V = V0
z = z0
# beta = x0[-1] # dRLなのでbeta推定．xの最後の要素をbetaとする
# beta = 0.99 # dRLなのでbeta推定．xの最後の要素をbetaとする

###### 効用関数 ######
def linkv(x): 
    vinst = np.exp(df_link['length']/10 * x[0]) 
    #+ df_link_integrated['staire'] * x[1]) # + df_link_integrated['staire_with_esc'] * x[3])
    return vinst # 出力形式はseries

###### 即時効用行列 ######
def Mset(x): 
    cost = linkv(x)
    cost = pd.concat([cost]*L, axis=1)
    cost = cost.T
    cost_numpy = cost.values # DataFrameをNumPy配列に変換
    return cost_numpy

###### 価値関数 ###### # 時間構造化するなら変わってくるが，，，
def newV(x):
    V = np.zeros((L, D))
    z = np.exp(V)
    
    for d in range(D): 
        d_linkid = ddata[d]
        
        d_index = df_link[df_link['linkid'] == d_linkid].index 
        z[d_index, d] = 1                   # 目的地がdの時，dでの期待効用Vd(d)=0から
        M = np.zeros((L, L))
        B = np.zeros((L, 1))
        B[d_index, 0] = 1

        # Mをdごとに更新？
        for k in range(L): 
            for a in range(L): 
                Ika = I[k, a] 
                if Ika == 1:                # 接続してなければ無関係（0のまま）
                    if a == d_index: 
                        M[k, a] = 1         # 吸収リンクの即時効用はexp(0)=1
                    else:
                        M[k, a] = Mset(x)[k, a]
        dL = 100
        zd = z[:, d].reshape(L, 1) 

        # z求解
        count = 0
        while dL >= 0.01: 
            zdd = zd.copy()
            zd = M @ (zdd ** x[-1]) + B 
            dL = np.sum(np.abs(zdd - zd)) 
            # dL = np.linalg.norm(zdd - zd, axis=0) 
            count += 1

        # z更新
        z[:, d] = zd[:, 0] 
        zd = np.where(zd == 0, 1, zd)  
        one_dim_array = np.log(zd).ravel()   # 1次元配列に
        V[:, d] = one_dim_array 

    return z

###### 対数尤度関数 ###### 
def loglikelihood(x):
    LL = 0

    # ここで観測モデルに入れる
    print(f'パラメタ{x}で観測モデルを回します')
    qall = measuring_model(x)
    # print(qall)

    condition = np.logical_or.reduce((qall[:, -1] == 16, qall[:, -1] == 20, qall[:, -1] == 21))

    # 条件に合致しない行のインデックスを取得します
    indices_to_keep = np.where(condition)[0]

    # 条件に合致しない行のみを抽出します
    filtered_qall = qall[indices_to_keep]

    # numpyのままではグルーピング不可なのでqallをdataframeにし，列カラムをつける
    mres = pd.DataFrame(filtered_qall, columns=['userid', 'timestep', 'maxbeacon', 'absorption']) # + [i for i in range(1, len(df_link_odds)*2+1)])

    # 観測結果をdで分割
    grouped = mres.groupby('absorption')
    df_list = [group for name, group in grouped]
    df_list = [data.reset_index(drop=True) for data in df_list]    
    # print(df_list)
    # loglikelihood内で価値関数更新
    Z = newV(x)

    # 目的地dごとに計算
    for d in range(D):
        d_linkid = ddata[d] 
        d_index = df_link[df_link['linkid'] == d_linkid].index
        d_index = d_index.tolist()
        d_index = d_index[0] 

        # 価値関数
        z = Z[:, d].reshape(L, 1) # 価値関数は固定じゃなくてloglikelihood内で逐次更新→直したらループが終わらないことは無くなった
        z = z ** x[-1]
        ZD = np.tile(z, (1, L))
        ZD = ZD.T

        # Mをdごとに更新（dループの外で問題はなさそう）
        M = np.zeros((L, L))
        for k in range(L): 
            for a in range(L): 
                Ika = I[k, a] 
                if Ika == 1:                            # 接続してなければ無関係（0のまま）
                    if a == d_index: 
                        M[k, a] = 1                     # 吸収リンクの即時効用はexp(0)=1（接続してたら）
                    else:
                        M[k, a] = Mset(x)[k, a]

        # 選択確率
        Mz = (M @ z != 0) * (M @ z) + (M @ z == 0)  
        MZ = np.tile(Mz, (1, L))  
        p = (M * ZD) / MZ

        # dを目的地とするユーザのデータを読み込む
        df = df_list[d]
        # print(df) # おかしくなってる．

        grouped2 = df.groupby('userid')
        df_list2 = [group for name, group in grouped2] # df_list2の長さが目的地をdとするuserの数に等しい
        df_list2 = [data2.reset_index(drop=True) for data2 in df_list2]
        count = 0

        for i in range(len(df_list2)): # 各userについてみて回る
            # print(f'{len(df_list2)}のなかで今{i}番目') # len(df_list2)が1になっている
            dfi = df_list2[i] # 各自の結果（userid, timestep, maxbeacon, absorption）
            userid = dfi.loc[0, 'userid']
            LLi = 0 # 個人iの尤度和
            at_cand_list = [] # 初期化
            # 各時刻ごとにmaxbeaconを参照．dfiの長さが滞在timestep数と同じになる
            for t in range(0, len(dfi)):                     # timestep1と最後とそれ以外で処理変わる
                max_beacon = int(dfi.loc[t, 'maxbeacon']) # intじゃないと後でスライスできないらしいわ
                ts = dfi.loc[t, 'timestep']
                print(f'user{userid}時刻{ts}でmaxbeaconは{max_beacon}')
                at_1_list = at_cand_list # at_1_listは膨れていくが，しょうがない？結局選択されなそうなリンクは尤度小さくなるだけなので，，
                at_cand_list = [] # 空にしておく

                if t == 0: # 初期のリンクは重要．max_beaconから，近いやつを候補にするが，幾つにするかで比較するか？とりあえず一個に決めるかー最短の．
                    # 次のmaxbeacon 
                    next_max_beacon = int(dfi.loc[1, 'maxbeacon'])
                    # next_max_beaconの座標
                    x_nb = df_ble.loc[next_max_beacon-1, 'x']
                    y_nb = df_ble.loc[next_max_beacon-1, 'y']
                    z_nb = df_ble.loc[next_max_beacon-1, 'floor']
                    loc_nb = np.array([x_nb, y_nb, z_nb])

                    # 候補リンク（場所は一意に決まって，シンプルに向きの関係で二つ候補ができる）
                    dist_list = d_array[:, max_beacon-1]
                    # print(dist_list)
                    dist_min_links = np.argwhere(dist_list == np.min(dist_list)).flatten()+1
                    # print(f'user{userid}時刻{ts}つまり最初のリンクの候補は{dist_min_links}')
                    dict2 = {}
                    for dist_min_link in dist_min_links:
                        dist_min_link_d = df_link[df_link['linkid'] == dist_min_link]['d'].iloc[0] # 注目リンクのdノード（のid）
                        # print(dist_min_link_d) # 値が二つ出ているのでエラーが出ている→ilocつけたら解消

                        x_mld = df_node[df_node['nodeid'] == dist_min_link_d]['x'] # そのdノードの各座標
                        y_mld = df_node[df_node['nodeid'] == dist_min_link_d]['y']
                        z_mld = df_node[df_node['nodeid'] == dist_min_link_d]['floor']
                        loc_minlinkd = np.array([x_mld, y_mld, z_mld])
                        
                        distance = np.linalg.norm(loc_minlinkd - loc_nb)
                        # print(dist_min_link, distance)
                        # linkidと次のスコア最大ビーコンとの距離を計算
                        dict2[dist_min_link] = distance
                    
                    # loc_minlinkdとloc_nbの距離が近い方を採用し，a0とする 
                    a0 = min(dict2, key = dict2.get) 
                    at_cand_list = [a0] # 初期は選択確率とかない．はず．一個に決めるなら
                    # print(f'user{userid}時刻{ts}つまり最初のリンクは{at_cand_list}．ちなみにこの時maxbeaconは{max_beacon}で次のbeaconは{next_max_beacon}')

                if t >= 1 and t <= (len(dfi)-2):            
                    for at_1 in at_1_list: # at_1に接続しているリンクを全部とる
                        a_row = I[at_1-1, :]
                        at_cand = np.where(a_row == 1)[0]+1 # np.whereを用いるidなので，+1をする．これlistのはず
                        at_cand_list.extend(list(set(at_cand) - set(at_cand_list))) # 重複なし．これが全atの候補
                    
                    # 各候補リンクに対してp(j|k)(pmat)を算出（これが観測確率になる）．kの候補がいっぱいあるので，まず全てのkに対して処理を行う
                    for at in at_cand_list:
                        dist = d_array[at-1, max_beacon-1]                                     
                        min_dist = d_array[:, max_beacon-1].min() # max_beaconに一番近いリンクとの距離
                        pmat = mq(dist, min_dist) # これは観測方程式，max_beaconと距離が近いリンクほど観測確率が大きくなるという単純な理屈
                        
                        # ここでもっとat_1を絞る．at_1_listに入っていてかつatと接続する→atの列が1である行のindex
                        at_col = I[:, at-1]
                        at_connected = np.where(at_col == 1)[0]+1 # np.whereを用いるidなので，+1をする．これlistのはず
                        # at_cand_list.extend(list(set(at_) - set(at_cand_list))) # 重複なし．これが全atの候補
                        common_list = [link for link in at_1_list if link in at_connected] # つまり

                        for at_1 in common_list:
                            # print(f'今見てるんはuser{userid}時刻{ts}max_beacon{max_beacon}，{at_1}から{at}への遷移')
                            at_prime_list = []
                            # 分子
                            # print(pmat, p[at_1-1, at-1])
                            nume = pmat * p[at_1-1, at-1]

                            # deno計算．at_1に接続しているリンクたち．
                            at_1_row = I[at_1-1, :]
                            at_prime_list = np.where(at_1_row == 1)[0]+1 # これリストのはず
                            # print(f'{at_1}に繋がっているリンクたち{at_prime_list}')
                            # at_prime_list.extend(list(set(at_prime) - set(at_prime_list))) # 重複するリスクないはずだが．

                            # 分母
                            deno = 0
                            for at_prime in at_prime_list:
                                # print(f'今のat_primeは{at_prime}')
                                dist = d_array[at_prime-1, max_beacon-1] # リンクの中点の方がいいかな→多分ね # ここもmax_beaconに対して実施（時刻tに紐づいている）
                                pmat_prime = mq(dist, min_dist)
                                deno += pmat_prime * p[at_1-1, at_prime-1]
                            if deno == 0:
                                continue
                            # print(f'{at_1}から{at}への遷移の時のdenoは{deno}')
                            delta = nume / deno # これがクロネッカーのデルタ

                            # 求めたら，これを元にLLitを決める．logzero回避
                            pp = (p[at_1-1, at-1] == 0) + ((p[at_1-1, at-1] != 0) * p[at_1-1, at-1])
                            LLi += delta * np.log(pp) # p=1になることはない，接続するのが一つでも滞在するという選択肢もあるので（のはず）
                            # print(f'↑の時，デルタ{nume}/{deno}={delta}，選択確率{pp}')
                                
                if t == (len(dfi) - 1):
                    # ここもatを一つに決める．最後のビーコンに応じたリンクをatに決め，(吸収リンクではなく最後のリンク→ではないポイか)
                    print(max_beacon)
                    aT = None
                    if max_beacon == 16:
                        aT = 99
                    if max_beacon == 20:
                        aT = 97
                    if max_beacon == 21:
                        aT = 98
                    print(aT)
                    if aT is not None:
                        dist = d_array[aT-1, max_beacon-1]
                        min_dist = d_array[:, max_beacon-1].min() # max_beaconに一番近いリンクとの距離
                        pmat = mq(dist, min_dist) # これは観測方程式，max_beaconと距離が近いリンクほど観測確率が大きくなるという単純な理屈
                        # print(f'最後のリンクは{aT}')
                        for at_1 in at_1_list:
                            at_prime_list = []
                            # 分子
                            nume = pmat * p[at_1-1, aT-1]
                            if nume == 0:
                                continue

                            # deno計算．今みているformer_link at-1に接続している全てのlinkに対して上記の計算を行う
                            # at_1に接続しているリンクたち．
                            at_1_row = I[at_1-1, :]
                            at_prime_list = np.where(at_1_row == 1)[0]+1
                            # at_prime_list.extend(list(set(at_prime) - set(at_prime_list)))

                            # 分母
                            deno = 0
                            for at_prime in at_prime_list:
                                dist_prime = d_array[at_prime-1, max_beacon-1] # リンクの中点の方がいいかな→多分ね # ここもmax_beaconに対して実施（時刻tに紐づいている）
                                pmat_prime = mq(dist_prime, min_dist) # min_distはt内で一定（t内でmaxbeaconが一定に決まるので）
                                deno += pmat_prime * p[at_1-1, at_prime-1]

                            if deno == 0:
                                    continue
                            
                            delta = nume / deno # これがクロネッカーのデルタ

                            # 求めたら，これを元にLLitを決める．logzero回避
                            pp = (p[at_1-1, aT-1] == 0) + ((p[at_1-1, aT-1] != 0) * p[at_1-1, aT-1])
                            LLi += delta * np.log(pp) # p=1になることはない，接続するのが一つでも滞在するという選択肢もあるので（のはず）
                            print(f'user{userid}時刻{ts}つまり最後のmaxビーコンのidは{max_beacon}，リンクは{aT}デルタは{nume}/{deno}で{delta}，pmatは{pmat}，経路選択確率は{pp}')   
            LL += LLi # 個人に対して和をとる
            count += 1

        print(f'パラメタ{x}のもとでLLが回り，計算したuserは{count}人でLL={LL}') # LLは負のはず
    
    return -LL



###### 推定部分 ###### 
dL = 100
n = 0

x_init = np.zeros(2)
bounds = [(-5, 5), (0, 1)] # , (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 1)]

# 注意．LLを返す関数frに対して数値微分を実行
def fr(x): 
    return -loglikelihood(x)

def hessian(x: np.array) -> np.array:
    h = 10 ** -4 # 数値微分用の微小量
    n = len(x)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.zeros(n), np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            
            res[i][j] = (fr(x + h * e_i + h * e_j)
            - fr(x + h * e_i - h * e_j)
            - fr(x - h * e_i + h * e_j)
            + fr(x - h * e_i - h * e_j)) / (4 * h * h)
    return res

def tval(x: np.array) -> np.array:
    print(f'hesse行列の逆行列{np.linalg.inv(hessian(x))}')            # inv逆行列を計算！
    print(f'各パラメタの分散{-np.diag(np.linalg.inv(hessian(x)))}')        # 対角成分取り出して-1倍
    return x / np.sqrt(-np.diag(np.linalg.inv(hessian(x))))

#num_random_initializations = 100
#for i in range(num_random_initializations):
    # x0をランダムにする
    #x0 = np.random.uniform(low=-10, high=10, size=(2,))  # Modify the range and size accordingly
    #print(f'{i+1}回目の乱数設定，x0は{x0}です')
dL = 100
    ######### 一旦whileループを外して初期値だけ変えて計算
while dL >= 0.01:
        print(f"<<<<<<<<<<<<<<<<now at {n}th loop in while cond>>>>>>>>>>>>>>>>>>")
        n += 1
        x = x0
            # V = V0 
            # z = z0 # 価値関数固定→結局Vはloglikelihood内で作ることにしたので与える必要がない

                # 構造推定part1：対数尤度関数最大化
        res = minimize(loglikelihood, x, method='L-BFGS-B', bounds = bounds) #, options={"maxiter":10,"return_all":True}) 
        
            # 推定値でパラメタxを更新
        x0 = res.x

        print("x0  =", x0)
        print("lnL =", -1*res.fun)

            # 構造推定part2：価値関数更新
            # z0 = newV(x0) 
            # zz0 = (z0 == 0) * 1 + (z0 != 0) * z0 
            # V0 = np.log(zz0)

                # 収束条件の計算（パラメタが収束しているか）
        dL = np.sum(np.abs(x - x0)) 
        print("dL =", dL)
        # print(n)
    #### 本来ここまでwhileループ

### hesse行列自作バージョン

hhh = hessian(x0)
print(hhh)
print(f'繰り返し回数{n}')
#tval = x0 / np.sqrt(np.diag(hhh))
tval = tval(x0)

### BFGS
"""
hhh = res.hess_inv #.todense()
    # print(n)
tval = x0 / np.sqrt(np.diag(hhh))
"""

L0 = -1 * loglikelihood(x_init) # 初期尤度
LL = -1 * res.fun

end_time = time.time()
proc_time = end_time - start_time

    ###### 最終結果の出力 ######
print("計算時間")
print(proc_time)
print("Inputdata")
print(search_folder)
print("結果の表示")
print(res)
    #print("説明変数の表示")
    #print(linkv)
    #print("NFXP")
print("初期尤度 = ", L0)
print("最終尤度 = ", LL)
print("ρ値 = ", (L0 - LL) / L0)
print("修正済ρ値 = ", (L0 - (LL - len(x0))) / L0)
print("パラメータ初期値 = ", x_init)
print("パラメータ推定値 = ", x0)
print("時間割引率 = ", x[-1])
print("t値 = ", tval)

