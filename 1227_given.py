## 情報量推定ver1226
## 観測分布・行動分布とも事象を条件付き分布に統一
## 二段階最適化，ぽい
## 価値関数求解は構造化を外した

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

############################
####### reading data #######
############################

df_data = pd.read_csv('/home/matsunaga/res2023/data/20230130_17/20230130_1745_18sec.csv')
search_folder = "/home/matsunaga/res2023/data/20230130_17/20230130_1745_18sec_end162021_under10_split/folder_6"
df_link = pd.read_csv('/home/matsunaga/res2023/data/nw/stanw_link_post.csv')
df_node = pd.read_csv('/home/matsunaga/res2023/data/nw/stanw_node.csv')
df_ble = pd.read_csv('/home/matsunaga/res2023/data/nw/ble_nw.csv')
file_list = list(os.listdir(search_folder))
L = len(df_link)


#########################################
####### 吸収リンク＆タイムステップ設定 #######
#########################################

T = 11 # 20ステップ以内ならT=21
ddata = [66, 67, 68] # 吸収リンク linkidで指定
D = len(ddata)

#######################
######### 準備 ########
#######################

###### 各linkのoとdの座標を入れておく配列 ###### 
link_loc_array = [] 
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
    z_d = df_node[df_node['nodeid'] == D]['floor'].iloc[0]
    d_loc = np.array([x_d, y_d, z_d])

    loc_tuple = (o_loc, d_loc)
    link_loc_array.append(loc_tuple) 


###### 各ビーコンの座標を入れておく配列 ###### 
beacon_loc_array = []
for j in range(len(df_ble)):
    beacon_id = df_ble.loc[j, 'ID']
    x_j = df_ble[df_ble['ID'] == beacon_id]['x'].iloc[0]
    y_j = df_ble[df_ble['ID'] == beacon_id]['y'].iloc[0]
    z_j = df_ble[df_ble['ID'] == beacon_id]['floor'].iloc[0]
    
    beacon_loc = np.array([x_j, y_j, z_j])
    beacon_loc_array.append(beacon_loc) 


###### ビーコンとリンク線分の中点間距離を返す関数 ######
def dist_to_mid(p1, p2, x):
    p1 = np.array(p1)
    p2 = np.array(p2)
    x = np.array(x)

    segment = p2 - p1
    v1 = x - p1
    v2 = x - p2
    mid_point = (p1 + p2) / 2

    v = v1 - np.dot(v1, segment) / np.dot(segment, segment) * segment # 垂直ベクトルを計算
    distance = np.linalg.norm(x - mid_point)
    
    return distance


###### d_arrayの用意 ###### # 各リンクの中点とbleビーコンとの距離配列
d_array = np.zeros((len(df_link), len(df_ble)))
for i in range(len(df_link)):
    p_o = link_loc_array[i][0] # o座標 # i=1の時
    p_d = link_loc_array[i][1] # d座標

    for j in range(len(df_ble)):
        x_ap = df_ble.loc[j, 'x']
        y_ap = df_ble.loc[j, 'y']
        z_ap = df_ble.loc[j, 'floor']
        p_ap = np.array([x_ap, y_ap, z_ap])

        d_array[i, j] = dist_to_mid(p_o, p_d, p_ap) # ijはlink i+1, ビーコンj+1の距離


###### 混雑度dens，maxRSSI，meanRSSI，検出回数を各ビーコン各timestepごとに入れた配列（ビーコン数23*タイムステップ数） ######
# 15分間のデータでやるならtimstep18secなら50step, 20secなら45step
dens_jt = np.zeros((23, 50), dtype = int)
max_jt = np.zeros((23, 50), dtype = int)
mean_jt = np.zeros((23, 50), dtype = int)
#count_jt = np.zeros((23, 45), dtype = int)

grouped = df_data.groupby(['ID', 'timestep']) # listにして渡す必要がある
df_list = [group.reset_index(drop=True) for name, group in grouped]

for group in df_list:
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


########################################
####### リンク接続行列＆プリズム準備 #######
########################################

def Imaking(link_data): 
    n = len(link_data)
    I = np.eye(n) # 全リンクに滞在リンク付与
    
    for i in range(n):

        if (i < 25) or (i >=  65): # 階段リンク以外の場合→これまで通り
            O = link_data.loc[i, 'o'] 
            D = link_data.loc[i, 'd'] 
            for j in range(n):
                if ((link_data.loc[j, 'o'] == O) or (link_data.loc[j, 'o'] == D)) or (link_data.loc[j, 'd'] == O) or (link_data.loc[j, 'd'] == D): 
                    I[i, j] = 1

        if 25 <= i < 65: ### iが25~64の場合→階段リンクの場合
            O = link_data.loc[i, 'o'] # onode
            D = link_data.loc[i, 'd'] # dnode

            for j in range(n):
                if link_data.loc[j, 'o'] == D: # DをOnodeとするリンクを1にしている
                    I[i, j] = 1
                
                if link_data.loc[j, 'o'] == D or link_data.loc[j, 'd'] == D: # odいずれかがDと一致するリンク[からの]遷移は0
                    I[j, i] = 0
                
                if link_data.loc[j, 'o'] == O or link_data.loc[j, 'd'] == O: # odいずれかがOと一致するリンクへの遷移は0
                    I[i, j] = 0

    return(I)

I = Imaking(df_link)

for i in range(26, 65, 2): # 階段には滞在リンク入れない
    I[i-1, i-1] = 0
    I[i, i] = 0

for i in range(25, 46): # 階段リンクに入ったら半分のところで折り返さない
    I[i, i+20] = 0
    I[i+20, i] = 0

for i in range(66, 69): # 吸収リンクからは吸収リンクにしか接続しない
    I[i-1, :] = 0 
    I[i-1, i-1] = 1 

# 起点Oからの到達可能性指示変数io
Itlist = np.zeros((T, L, L)) ## これは各状態に対して与えるので遷移回数T-1よりひとつ多くてT成分考える（timestep数がTなら状態数はT）
II = np.copy(I)
Itlist[0, :, :] = np.eye(L) 

for ts in range(1, T): 
    Itlist[ts, :, :] = II 
    II = np.dot(II, I)
    II = np.where(II > 0, 1, 0) 

# 終点Dからの到達可能性指示変数id ## これもIlist同様の考え方
Ittlist = np.zeros((T, L, L))
for ts in range(T): 
    Ittlist[ts, :, :] = np.transpose(Itlist[T - ts - 1, :, :]) 

#####################
#### od_list作成 #### # ビーコン列からODを特定（ちょっとよくないけどとりあえず）
#####################

saikyo_key = [1, 2, 3, 4, 5, 6]
yamate_key = [7, 8, 9, 10, 11, 12, 13]
discriminate_key = [19, 22, 23]
abs_key = [16, 20, 21]

od_list = pd.DataFrame(columns=['userid', 'o', 'd', 'abs', 'Ti'])

count_file = 0
for file_name in file_list:

    dfi = pd.read_csv(os.path.join(search_folder, file_name))   # file_path = os.path.join(search_folder, file_name)
    userid = int(file_name.split('.')[0])                       # userid = int(base_name)

    Ti = dfi['timestep'].nunique()                              # 個人の滞在時間 Tiまでにlast_linkにくる   
    grouped = dfi.groupby('timestep')
    dfi_list = [group.reset_index(drop=True) for name, group in grouped]  

    ### 各人の各時刻の最大スコアのbeacon列 ###
    max_beacon_list = [] 
    for t in range(Ti):
        dfit = dfi_list[t] # 時刻tの全ログデータ
        ts = int(dfit.loc[0, 'timestep'])

        captured_beacon = dfit.groupby('ID')
        captured_beacon_list = [group.reset_index(drop=True) for name, group in captured_beacon]
        tdict = {}

        for j in range(len(dfit)):
            jid = dfit.loc[j, 'ID'] # そのログデータのビーコン
            max_rssi = max_jt[jid-1, ts-1] # そのビーコンの時刻tsでのmax_rssi
            rssi = dfit.loc[j, 'RSSI'] # そのログの電波強度
            jscore = max_rssi / rssi # このログのスコア→これを辞書のjidのところに足すイメージ
            if jid not in tdict:
                tdict[jid] = 0
            tdict[jid] += jscore  

        jmax = max(tdict, key = tdict.get) 
        max_beacon_list.append(jmax) #### なんかここ空になってる，なんで→append使う時はa = a.append()ってしないでa.append()だけにするらしいわ
        # これはGPTに言われるまで知らなかった，，，

    ### Oの確定 ### 
    first_beacon = max_beacon_list[0]
    first_dist_list = d_array[:, first_beacon-1] 
    first_dist_dict = {index + 1: value for index, value in enumerate(first_dist_list)} # keyはlinkidになっている．max_beaconと各リンクの距離が入ってる

    if any(item in discriminate_key for item in max_beacon_list): # 埼京オンリーにする
        filtered_dict = {key: first_dist_dict[key] for key in first_dist_dict if key in saikyo_key}
        max_key = min(filtered_dict, key = filtered_dict.get) # これが初期リンクとなる．埼京ホームのうち，最もmax_beaconに近いリンク
    else:
        filtered_dict = {key: first_dist_dict[key] for key in first_dist_dict if key in yamate_key}
        max_key = min(filtered_dict, key = filtered_dict.get)

    a0 = int(max_key)       # linkid
        
    #### Dの確定 #### 
    last_beacon = max_beacon_list[-1]

    if last_beacon not in abs_key: 
        continue
        
    last_link = 0
    if last_beacon == 16: # node39
        last_link = 25 
        abs_link = 46
    elif last_beacon == 20: # node28
        last_link = 17
        abs_link = 47
    elif last_beacon == 21: # node29
        last_link = 18   
        abs_link = 48   
    else:
        continue

    ## ここまで条件を満たせばod_listにデータを追加
    od_list.loc[count_file, 'userid'] = userid
    od_list.loc[count_file, 'o'] = a0
    od_list.loc[count_file, 'd'] = last_link
    od_list.loc[count_file, 'abs'] = abs_link
    od_list.loc[count_file, 'Ti'] = Ti

    count_file += 1

OD = len(od_list)


##########################
###### 時空間プリズム ######
##########################

Ilist = []
for i in range(OD):   

    # OD確定読み込み
    ao = od_list.loc[i, 'o'] - 1  # index
    ad = od_list.loc[i, 'd'] - 1  # index
    aabs = od_list.loc[i, 'abs'] - 1
    userid = od_list.loc[i, 'userid']

    Id = np.zeros((T-1, L, L)) # Id[t]は時刻tの時の利用可能遷移k→aを示す．最後のt=T-1のとき不要なので要素数はt=0~T-2のT-1個

    for ts in range(T-1): # Tまでに吸収されるのでdはaabs．遷移回数はT-1なのでこれでOK
        if ts == 0:
            Id[ts, ao, :] = I[ao, :] 
            continue
        
        alist = np.where(Itlist[ts + 1, ao, :] == Ittlist[ts + 1, aabs, :])[0] ## ts=1（つまり第二回目の状態）のとき3番目の状態を見ている→OK
        ## Itlistの長さはT．最後はT-2で，そのときT-2+1=T-1, 
        for a in alist:
            if Itlist[ts + 1, ao, a] == Ittlist[ts + 1, aabs, a]: # always True
                klist = np.where(I[:, a] == 1)[0]

                for k in klist:
                    if len(np.where(Id[ts - 1, :, k] == 1)[0]) != 0:
                        Id[ts, k, a] = 1 

    Ilist.append(Id)


#######################
###### 観測モデル #######
#######################

def qjx(rssi, max_rssi, x_loc, j, b):               # bは観測誤差分散のパラメータ．x_loc（x, y, z)にいたときビーコンjでrssiの観測が得られる確率
    j_loc = beacon_loc_array[j-1]                   # id=jのビーコンの位置座標
    #s = max_rssi
    d_est = 10 ** ((max_rssi - rssi)/(10*b[0]))     # 減衰パラメタb[0] rssi0→-40にしてみる
    d = np.linalg.norm(x_loc - j_loc) 
    f = norm.pdf(d, d_est, b[1])                    # 誤差分散b[1]
    omega = (100 + rssi) / 10 

    return omega * f

# -----****------とりあえず定義はしておく．MmodelはxとbからQの分布を与える-----****-----
def Mmodel(x, b):

    Pall = newPall(x)

    count = 0 
    
    SLlist = [] 
    TLlist = []

    for od in range(OD):

        # odに対応するuserのファイルを読み込む（od_listのuserid列が実データとod_listを紐づけている）
        userid = od_list.loc[od, 'userid']
        file_name = f'{userid}.csv'
        dfi = pd.read_csv(os.path.join(search_folder, file_name)) 
        grouped = dfi.groupby('timestep')
        dfi_list = [group.reset_index(drop=True) for name, group in grouped]  

        Ti = od_list.loc[od, 'Ti']              # 個人odの観測限界．時刻Tiでlast_linkに到達し，時刻Ti+1で吸収される
        ao = od_list.loc[od, 'o'] - 1           # index
        ad = od_list.loc[od, 'd'] - 1           # index
        aabs = od_list.loc[od, 'abs'] - 1

        #### プリズム用意 #### odを与えたらプリズム制約．時刻Ti+1つまりTi回の遷移でaoからabsまで行く．もしくは時刻TiつまりTi-1回の遷移でaoからlastまで行く
        Idm = np.zeros((Ti, L, L)) ## 1成分目の値が遷移回数．Tiにするならabsまで行くことを仮定している．そのときItimlistは状態数＝時間数なのでTi+1で良い
        Itmlist = np.zeros((Ti+1, L, L))
        IIm = np.copy(I)
        Itmlist[0, :, :] = np.eye(L)

        for tt in range(1, Ti+1): 
            Itmlist[tt, :, :] = IIm 
            IIm = np.dot(IIm, I) 
            IIm = np.where(IIm > 0, 1, 0)

        Ittmlist = np.zeros((Ti+1, L, L))
        for tt in range(Ti+1): 
            Ittmlist[tt, :, :] = np.transpose(Itmlist[(Ti+1) - tt - 1, :, :])

        for tt in range(Ti):
            if tt == 0:
                Idm[tt, ao, :] = I[ao, :] 
                continue

            alist = np.where(Itmlist[tt + 1, ao, :] == Ittmlist[tt + 1, aabs, :])[0] # ここadじゃなくてaabsが適切では????
            for a in alist:
                if Itmlist[tt + 1, ao, a] == Ittmlist[tt + 1, aabs, a]: # ここadじゃなくてaabsが適切では????
                    klist = np.where(I[:, a] == 1)[0] 

                    for k in klist:
                        if len(np.where(Idm[tt - 1, :, k] == 1)[0]) != 0:
                            Idm[tt, k, a] = 1 ### 時刻tにおいて，a(t)→a(t+1)への遷移がOKなところが1になっている                            

        #SLmatrix = np.zeros((T, L)) # status_likelihood 状態数は遷移数+1
        TLmatrix = np.zeros((T-1, L, L)) # trans likelihood
        pmmatrix = np.zeros((T-1, L, L))  # p(m|a)のmatrixのはず
        pmstmatrix = np.zeros((T-1, L, L)) # 同時発生尤度のマトリックス
        #SLmatrix[0, ao] = 1 # t = 0ではaoに確定的に存在するとする ####### ←ここがまずよくない．
        klist = [ao]
        for t in range(1, T): #### t=tでは，k=a(t-1)からa=a(t)への遷移を考える．TLmatrixは遷移数＝T-1個しかないのでindexは0 ~ t-2

            #######################################
            ######### 1. 遷移発生尤度を計算 ##########
            #######################################

            if t <= Ti - 1:

                dfit = dfi_list[t] 
                ts = int(dfit.loc[0, 'timestep'])

                #slsum = 0
                #nlist = np.argsort(SLmatrix[t-1])[::-1]  # tsにおける状態尤度が大きい順に並び替えたインデックスリスト
                ## t=1のときSL[0]を参照するがaoしかないのでそのままaoを見ることになる．t=2のときSL[1]を参照，tのときt-1を参照
                ## ループは1からまでにしてt-1における尤度が大きいリンクに対して??実装?
                ## t-1で尤度大きいリンクをkとしてtのリンクをaとして扱った方が実態に近い

                #klist = [] # つまり時刻t-1で状態尤度が高い上位のリンクをklistに入れている

                #for i in range(L):
                #    slsum += SLmatrix[t-1, nlist[i]]
                #    klist.append(nlist[i])
                #    if slsum >= 0.9:
                #        break
                
                #klist3 = [i+1 for i in klist]
                
                #print(f'時刻{t-1}での状態発生尤度の高かったリンクは{klist3}') ##

                for k in klist: ## t-1での滞在尤度が高いリンクに対して走査

                    for a in range(L): ## tのリンクはこっち

                        if Idm[t-1, k, a] == 0: ### t-1時点でk→aが利用可能か否かを見る ###### 
                            TLmatrix[t-1, k, a] = 0 ### 多分これでいいのか？
                            continue
                        
                        a_loc = link_loc_array[a]
                        a_o = a_loc[0]
                        a_d = a_loc[1]

                        ## 位置については一旦保留→aの上に確率的に存在するという仮定が望ましいが一旦リンクの中点にいるとする
                        x_loc = (a_o + a_d) / 2

                        pm = 1

                        for j in range(len(dfit)): ### ここでようやく各観測ログにアプローチ
                            ### rssi小さい観測がpmの極小化を招いている説もある，積をとっているので．
                            ### でも相対評価で尤度最大化しようとしているならそこまで気にならないはずだが
                            j_id = dfit.loc[j, 'ID']
                            j_rssi = dfit.loc[j, 'RSSI']
                            j_max_rssi = max_jt[j_id - 1, ts - 1] 

                            pm *= qjx(j_rssi, j_max_rssi, x_loc, j_id, b) # 観測方程式
                        
                        pmmatrix[t-1, k, a] = pm # 時刻tの観測尤度はリンクaの観測確率ね→ただしindexの関係でt-1に入れる．つまりt=1にat-1=a0=k→at=a1=aの観測尤度がpm[0]に入ってる
                        #print(f'時刻{t}でリンクa(t)つまり{a+1}の観測尤度{pm}')
                
                # 遷移尤度のnume pmst 時刻tにおいてk(t-1)→a(t)への遷移が観測された確率がt-1に入ってる
                # 時刻tにおいてk(t-1)→a(t)への遷移確率がPall[t-1]に入っている=時刻t-1にk(t-1)→a(t)に遷移する確率がPall[t-1]ということ
                # つまり時刻tにおけるat→at+1への遷移確率はPall[t]に入っているということか．で，aT-1→aTへの遷移確率まで入っているがaT-2→aT-1までしか使わない
                pmstmatrix[t-1, :, :] = pmmatrix[t-1, :, :] * Pall[od, t-1, :, :] # 観測確率pm*遷移確率（正規化前の観測尤度）

                # 正規化
                deno = np.sum(pmstmatrix[t-1, :, :], axis=1) # np.sum(pmstmatrix[t-1], axis=1)になってた
                deno = (deno == 0) * 1 + (deno != 0) * deno ## 観測尤度の分母
                
                deno_rep = np.tile(deno, L).reshape(L, L).T ##### これがないと変になった

                pmstmatrix[t-1, :, :] = pmstmatrix[t-1, :, :] / deno_rep ## これが観測尤度

                #sl = SLmatrix[t-1, :] # このままでは行ベクトル
                #sl_col = sl[:, None] #### ここがRと文法異なる部分

                TLmatrix[t-1, :, :] = pmstmatrix[t-1, :, :] # * sl_col # 同時発生尤度を計算してる（SLmatrixはindexが1増えてるので，同じt-1でもひとつ前の段階を指名している
                # 時刻tにk(a_t-1)からa(a_t)への遷移が発生する尤度

                # 非ゼロ要素のインデックスを取得
                # non_zero_indices = np.transpose(np.nonzero(TLmatrix[t-1, :, :])) +1

                # タプル形式のリストに変換
                # non_zero_tuples = [tuple(index) for index in non_zero_indices]

                #### klist更新
                klist = []
                TLt = TLmatrix[t-1, :, :]
                # 方法1: any() を使って非零の要素を持つ列を見つける
                non_zero_columns = np.any(TLt != 0, axis=0)
                klist = np.where(non_zero_columns)[0]

                # 方法2: nonzero() を使って非零の要素がある列のインデックスを取得する
                non_zero_indices = np.nonzero(np.any(TLt != 0, axis=0))[0]
                klist = non_zero_indices.tolist()

            elif t == Ti:
                TLmatrix[t-1, ad, aabs] = 1 ### それ以外はそのまま→0のまま
            
            else:
                TLmatrix[t-1, aabs, aabs] = 1 ### それ以外はそのまま→0のまま

            ########################################
            ########## 2. 状態発生尤度を定義 ##########
            ########################################
            """
            if t <= Ti - 2: # 観測プリズム（ao, ad, Tiから決定）Idm内の状態しか取り得ない．k(a_t-1)からa(a_t)への遷移がOKのときIdmt[t-1]が1

                dfit = dfi_list[t] # t<len(dfit)=Tiにおいて指定可能
                ts = int(dfit.loc[0, 'timestep'])

                for i in range(L): ## iは遷移先リンク，つまりa(t)の候補ということになる

                    klist = np.where(Idm[t-1, :, i] == 1)[0] ## 前時刻参照．時刻tにおいてリンクi=a(t)に遷移可能な前リンクk=a(t-1)を探し，klistに入れている．この時，構造化NWと観測情報からリンク候補はかなり絞られている

                    if len(klist) == 0:
                        continue

                    kTL = 0
                    for k in klist: # このiに移動できる元リンク（t-1時点でのリンクa(t-1)）
                        kTL += TLmatrix[t-1, k, i] ## 時刻t-1にk=a(t-1)からi = a(t)への遷移発生尤度をkTLに足していき，これがtでのiの存在尤度になる（それはそう！！）
                        #print(f'状態遷移 k = a(t-1) = {k+1} → a = a(t) = {i+1}（全部リンクid）の遷移発生尤度は{kTL}')

                    SLmatrix[t, i] = kTL ### SLmatrixはindex t に入れる（SL[0]を事前に定義済みなので）

            if t == Ti - 1: # この時刻で最終リンクに確定的に滞在
                #print(f'時刻{t}で最後のリンク')
                SLmatrix[t, ad] = 1

            elif t >= Ti: # この時刻以降は確定的に吸収に滞在
                #print(f'時刻{t}で吸収リンク')             
                SLmatrix[t, aabs] = 1
            """
        #SLlist.append(SLmatrix)
        TLlist.append(TLmatrix)
    
    return(TLlist) 



###########################
####### 経路選択モデル ######
###########################

###### 即時効用行列 ###### # 混雑内生性考えないならT次元は圧縮可能だが一応→時不変だし，経路長も一緒だし経路パラメタしか入れてないので全リンクの効用が等しくなるよ．
def Mset(x): 
    #Probs_T = Probs.T # Probsを読み込む（内生性バージョン）
    inst = np.zeros((T-1, L, L))
    for t in range(T-1):
        inst_t = np.exp(df_link['length']/10 * x[0]) # + Probs_T[:, t] * x[1])    # + df_link['staire'] * x[1]) # + df_link_integrated['staire_with_esc'] * x[3])
        inst_t = pd.concat([inst_t]*L, axis=1) 
        inst_t = inst_t.T
        inst_t_numpy = inst_t.values # DataFrameをNumPy配列に変換
        inst[t, :, :] = inst_t_numpy

    return inst


###### 価値関数更新 ###### # 状態価値関数なので状態の数＝Tだけある
"""
def newV(x):
    beta = x[-1]
    z = np.ones((T, L, OD)) # なんでODが第三成分なのかがよくわからないが
    V = np.zeros((T, L, OD))

    for od in range(OD):
        # print(od_list.loc[od, 'userid'])
        Id = Ilist[od]

        ##### 即時効用 #####
        M = np.zeros((T-1, L, L))
        for ts in range(T-1):
            Mts = Id[ts, :, :] * Mset(x)[ts, :, :] # 内生性考慮してないのでMsetは定常
            M[ts, :, :] = Mts

        ##### 価値関数 ##### 
        for t in range(T-1, 0, -1): # T-1から1までのT-1回
            zi = np.dot(M[t-1, :, :], (z[t, :, od] ** beta) )
            z[t-1, :, od] = (zi==0)*1 + (zi!=0)*zi
            V[t-1, :, od] = np.log(z[t-1, :, od])

    return z
"""

###### リンク選択確率計算 ######
def newPall(x):
    Pall = np.zeros((OD, T-1, L, L)) # 個人ごと，時刻ごとの各リンク間遷移確率行列 遷移の回数なのでT-1．今回Tを状態の数としているので遷移数はT-1になる
    beta = x[-1]

    #### 以降個人ごとに処理 #### 
    for od in range(OD): 
        
        Id = Ilist[od] # すでに個人のプリズムはIlistで用意してある

        ##### 即時効用 #####
        M = np.zeros((T-1, L, L))
        for ts in range(T-1):
            Mts = Id[ts, :, :] * Mset(x)[ts, :, :] # 内生性考慮してないのでMsetは定常
            M[ts, :, :] = Mts
            
        ##### 価値関数 ##### ### ここで価値関数求解（newVに相当）が入っている
        ##### 価値関数別で計算するならここにVを外から読み込むことになる
            
        z = np.ones((T, L))
        for t in range(T-1, 0, -1):
            ####
            zii = M[t-1, :, :] * (z[t, :] ** beta) 
            zi = zii.sum(axis = 1)
            z[t-1, :] = (zi==0)*1 + (zi!=0)*zi

        ##### 選択確率行列 ##### # 尤度計算は不要
        for t in range(T-1): # 選択確率は各時刻tごとに決まる
            for k in range(L):
                for a in range(L):
                    if M[t, k, a] == 0: # 接続条件を満たせなかった観測は排除（logzero回避）
                        continue # ここがbreakになってたのが癌だった
            
                    Pall[od, t, k, a] += np.exp(np.log(M[t, k, a]) + beta * np.log(z[t+1, a]) - np.log(z[t, k])) 

    return Pall # 個人ごとの各時刻ごとのリンク遷移確率の一覧がPallに入った


#### Pを動かす場合の目的関数（m-step）
def KLP(x): 
    KLP = 0
    Pt = newPall(x)

    ## 観測モデルQは固定
    TLlist = Mmodel(y, b0) ### ここのx, bは固定→ややこしいのでyでおく．y=x(k-1)

    ## 選択モデルPは動かすのでここのxは動かす
    #pjoint = Pjoint(x) # givenはこれは不要

    for od in range(OD):
        userid = int(od_list.loc[od, 'userid'])
        TLmatrix = TLlist[od]
        for t in range(T-1):
            KLPit = 0
            pit = Pt[od, t, :, :] # 時刻tにk(a_t-1)とa(a_t)の同時滞在確率はpjoint[t-1]に入っている
            pit = (pit == 0) * 1 + (pit != 0) * pit

            for k in range(L): # 時刻tにa(t-1)からa(tへの遷移を考える)
                for a in range(L):
                    logpka = 0
                    ## 遷移確率ではなく同時滞在確率と同時滞在尤度を比較するという構図にする
                    if pit[k, a] == 0:
                        logpka = 0
                        continue
                    
                    if TLmatrix[t, k, a] == 0:
                        continue

                    else:
                        p = pit[k, a] # 同時滞在確率
                        q = TLmatrix[t, k, a] # 同時滞在尤度
                        #print(f'個人{userid}時刻{t}の{k+1}と{a+1}の同時滞在確率は{p}，同時発生尤度は{q}')
                    qlp = q * np.log(p)
                    qlq = q * np.log(q)

                    # KLPit += (qlq - qlp)
                    KLPit += qlp # KLではなくなく交差エントロピー（対象事象が揃ったのでどっちもで計算できるはず）
            
            KLP += KLPit
            #print(f'user{userid}，時刻{t}の一般化平均情報量{-KLit}（正，小さい方がいい）')

    print(f'モデルパラメタ{x}でのクロスエントロピー{-KLP}')

    return -KLP


#### Qを動かす場合の目的関数（e-stepもどき）
def KLQ(b): 
    KLQ = 0

    ## Q動かすが，bのみ動かしxはここでも固定，つまりQ内のxは常に所与の値として，Qはbのみの関数として扱う
    TLlist = Mmodel(y, b)

    ## Pは固定するので前に推定したx0に固定する
    # pjoint = Pjoint(y)
    Pt = newPall(y)

    for od in range(OD):
        userid = int(od_list.loc[od, 'userid'])
        TLmatrix = TLlist[od]
        for t in range(T-1):
            KLQit = 0
            pit = Pt[od, t, :, :] # 時刻tにk(a_t-1)とa(a_t)の同時滞在確率はpjoint[t-1]に入っている
            pit = (pit == 0) * 1 + (pit != 0) * pit

            for k in range(L): # 時刻tにa(t-1)からa(tへの遷移を考える)
                for a in range(L):
                    logpka = 0
                    ## 遷移確率ではなく同時滞在確率と同時滞在尤度を比較するという構図にする
                    if pit[k, a] == 0:
                        logpka = 0
                        continue
                    if TLmatrix[t, k, a] == 0:
                        continue
                    else:
                        p = pit[k, a] # 同時滞在確率
                        q = TLmatrix[t, k, a] # 同時滞在尤度
                        #print(f'個人{userid}時刻{t}の{k+1}と{a+1}の同時滞在確率は{p}，同時発生尤度は{q}')
                    qlp = q * np.log(p)
                    qlq = q * np.log(q)
                    KLQit += qlp # KLではなくなく交差エントロピー（対象事象が揃ったのでどっちもで計算できるはず）
            KLQ += KLQit

    print(f'観測パラメタ{b}でのクロスエントロピー{-KLQ}')

    return -KLQ


#####################
###### 推定部分 ###### 
#####################

## 
def frP(x): 
    return -KLP(x)

def frQ(b):
    return -KLQ(b)

def hessianP(x: np.array) -> np.array:
    h = 10 ** -4 # 数値微分用の微小量
    n = len(x)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.zeros(n), np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            
            res[i][j] = (frP(x + h * e_i + h * e_j)
            - frP(x + h * e_i - h * e_j)
            - frP(x - h * e_i + h * e_j)
            + frP(x - h * e_i - h * e_j)) / (4 * h * h)

    return res

def tvalP(x: np.array) -> np.array:
    print(f'hesse行列の逆行列{np.linalg.inv(hessianP(x))}')         
    print(f'各パラメタの分散{-np.diag(np.linalg.inv(hessianP(x)))}')      
    return x / np.sqrt(-np.diag(np.linalg.inv(hessianP(x))))

def hessianQ(b: np.array) -> np.array:
    h = 10 ** -4 # 数値微分用の微小量
    n = len(b)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.zeros(n), np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            
            res[i][j] = (frQ(b + h * e_i + h * e_j)
            - frQ(b + h * e_i - h * e_j)
            - frQ(b - h * e_i + h * e_j)
            + frQ(b - h * e_i - h * e_j)) / (4 * h * h)

    return res

def tvalQ(b: np.array) -> np.array:
    print(f'hesse行列の逆行列{np.linalg.inv(hessianQ(b))}')         
    print(f'各パラメタの分散{-np.diag(np.linalg.inv(hessianQ(b)))}')      
    return x / np.sqrt(-np.diag(np.linalg.inv(hessianQ(b))))


## パラメタ初期設定 ##
dL = 100
n = 0

xdim = 2
bdim = 2
x0 = np.zeros(xdim)
b0 = np.zeros(bdim) 
x_init = np.zeros(xdim)
b_init = np.ones(bdim)

x0[0] = -1.5
x0[-1] = 0.8
b0[0] = 2
b0[1] = 15
b_init[0] = 2
b_init[1] = 15
 
bbounds = [(2, 4), (10, 35)]
xbounds = [(-10, -0.01), (0, 1)] #, (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 1)]
print('今からwhile')

while dL >= 1:
    print(f"<<<<<<<<<<<<<<<<now at {n}th loop in while cond>>>>>>>>>>>>>>>>>>")
    n += 1
    y = x0
    b = b0

    # e-step
    resb = minimize(KLQ, b, method='L-BFGS-B', bounds = bbounds) #, options={"maxiter":10,"return_all":True}) 
    b0 = resb.x
    print("b0  =", b0)
    print("lnKLQ =", -1*resb.fun)

    x = x0

    # m-step
    resx = minimize(KLP, x, method='L-BFGS-B', bounds = xbounds)
    x0 = resx.x
    print("x0  =", x0)
    print("lnKLP =", -1*resb.fun)

    dL = np.sum(np.abs(b - b0)) + np.sum(np.abs(x - x0)) 
    print("dL =", dL)

print('while文終わり')

### hesse行列自作バージョン
hhhP = hessianP(x0)
tvalP = tvalP(x0)

hhhQ = hessianQ(b0)
tvalQ = tvalQ(b0)


### BFGS
"""
hhh = res.hess_inv #.todense()
    # print(n)
tval = x0 / np.sqrt(np.diag(hhh))
"""
"""
z_init = newV(x_init)
zz_init = (z_init==0)*1 + (z!=0)*z
V = np.log(zz_init)
"""

# while文で更新済みのy, b, xを使う
KLP_res = -1 * resx.fun

# while文に関係のないinitを用いたy, b
y = x_init ## これ怪しい，評価相手が曖昧
b0 = b_init
#KLP0 = -1 * KLP(x_init) # 初期尤度（情報量）

end_time = time.time()
proc_time = end_time - start_time

###### 最終結果の出力 ######
#print("計算時間")
#print(proc_time)
print("Inputdata")
print(search_folder)
print("結果の表示")
print(resx)
print(resb)
#print("説明変数の表示")
#print(linkv)
print("モデルパラメータ初期値 = ", x_init)
print("観測パラメータ初期値 = ", b_init)
print("モデルパラメータ推定値 = ", x0)
print("観測パラメータ推定値 = ", b) # b0にはinitが入ってしまうので
print("時間割引率 = ", x0[-1])
print("モデルパラメータt値 = ", tvalP)

print("観測パラメータt値 = ", tvalQ)

KLP0 = -1 * KLP(x_init) # 初期尤度（情報量）

print("初期尤度 = ", KLP0)
print("最終尤度 = ", KLP_res)
print("ρ値 = ", (KLP0 - KLP_res) / KLP0)
print("修正済ρ値 = ", (KLP0 - (KLP_res - len(x0))) / KLP0)


end_time = time.time()
proc_time = end_time - start_time
print("計算時間")
print(proc_time)