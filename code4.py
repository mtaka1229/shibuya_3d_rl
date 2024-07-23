# 観測方程式の考え方が全くわかってなかったので一からやり直す
# Dを絞って実行

##### 同時推定コード #####
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

start_time = time.time()
print('start!')

####### reading data #######
df_quater = pd.read_csv('/home/matsunaga/res2023/data/20230130_08/20230130_0815_kai.csv')

df_link_odds = pd.read_csv("/home/matsunaga/res2023/data/nw/shibuya_link_1112/shibuya_link_post_odds.csv")
l = len(df_link_odds)

df_link_integrated = pd.read_csv('/home/matsunaga/res2023/data/nw/shibuya_link_post_integrated?.csv')
L = len(df_link_integrated)

search_folder = "/home/matsunaga/res2023/data/20230130_08/user_stock_kai_end16"
file_list = list(os.listdir(search_folder))

df_node = pd.read_csv('/home/matsunaga/res2023/data/nw/shibuya_node_corrected.csv')
df_ble = pd.read_csv('/home/matsunaga/res2023/data/nw/ble_nw.csv')

###### 各linkのoとdの座標を入れておく配列 ###### # 高さ方向の検出を排除（床の遮蔽とか考え出すとやばいので）
oddslink_loc_array = [] 
for i in range(len(df_link_odds)):
    linkid = df_link_odds.loc[i, 'linkid']
    O = df_link_odds.loc[i, 'O']
    D = df_link_odds.loc[i, 'D']

    x_o = df_node[df_node['nodeid'] == O]['x'].iloc[0]
    y_o = df_node[df_node['nodeid'] == O]['y'].iloc[0]
    z_o = df_node[df_node['nodeid'] == O]['floor'].iloc[0] * 1000
    o_loc = np.array([x_o, y_o, z_o])

    x_d = df_node[df_node['nodeid'] == D]['x'].iloc[0]
    y_d = df_node[df_node['nodeid'] == D]['y'].iloc[0]
    z_d = df_node[df_node['nodeid'] == D]['floor'].iloc[0] * 1000 # これでseriesではなく値として抽出できるはず
    d_loc = np.array([x_d, y_d, z_d])

    loc_tuple = (o_loc, d_loc)

    oddslink_loc_array.append(loc_tuple) # oddslink_loc_array[linkid//2][0]がlinkidのonode, [1]がdnode，[2]が中点，，，，

###### ビーコンとリンク線分の最短距離返す関数 ######
def shortest_distance_to_segment(p1, p2, x):
    p1 = np.array(p1)
    p2 = np.array(p2)
    x = np.array(x)
    
    segment = p2 - p1
    v1 = x - p1
    v2 = x - p2
    
    # 垂直ベクトルを計算
    v = v1 - np.dot(v1, segment) / np.dot(segment, segment) * segment
    
    # 垂直ベクトルのノルムを求める
    distance = np.linalg.norm(v)
    
    # 線分の範囲内であるかチェック
    dot_product = np.dot(v1, segment)
    if dot_product < 0:
        return np.linalg.norm(x - p1)
    elif dot_product > np.dot(segment, segment):
        return np.linalg.norm(x - p2)
    else:
        return distance

###### d_arrayの用意 ###### # 各リンク(奇数側)とbleビーコンとの距離配列
d_array = np.zeros((len(df_link_odds), len(df_ble)))

for i in range(len(df_link_odds)):
    p_o = oddslink_loc_array[i][0] # o座標 # i=1の時
    p_d = oddslink_loc_array[i][1] # d座標

    for j in range(len(df_ble)):
        x_ap = df_ble.loc[j, 'x']
        y_ap = df_ble.loc[j, 'y']
        z_ap = df_ble.loc[j, 'floor']*1000 # zの値をめちゃ大きくすることで，階を挟んだ電波捕捉を捨象（階の厳密な高さが不明なのと，床板・天井版による電波遮蔽を考慮するため）．階が同じなら0になるので変な影響はない
        p_ap = np.array([x_ap, y_ap, z_ap])

        d_array[i, j] = shortest_distance_to_segment(p_o, p_d, p_ap) # ijはlink 2i+1, ビーコンjの距離

###### 各ビーコンからxm圏内のリンクリスト ######
linklist_near_j = []
for j in range(len(df_ble)):
    column_j = d_array[:, j]
    link_list = np.where(column_j <= 10)[0] # 10m以下の場所の指定（インデックスが得られる）
    link_list = [idx*2 + 1 for idx in link_list] # link_listの各要素はindexなので，2掛けて1を足せばlinkidになる

    linklist_near_j.append(link_list)

###### ノード間距離の配列 ######
d_node_array = np.zeros((len(df_node), len(df_node))) # d_node_arrayのi行j列がnode i+1とnode j+1の間の距離となるように計算
for i in range(len(df_node)):
    xi = df_node.loc[i, 'x']
    yi = df_node.loc[i, 'y']
    zi = df_node.loc[i, 'floor']
    i_loc = np.array([xi, yi, zi])
    
    for j in range(len(df_node)):
        xj = df_node.loc[j, 'x']
        yj = df_node.loc[j, 'y']
        zj = df_node.loc[j, 'floor']
        j_loc = np.array([xj, yj, zj])

        d_node_array[i, j] = np.linalg.norm(i_loc - j_loc)

###### ビーコンとノード間の距離の配列 ###### # ビーコンi+1とnode j+1間の距離
d_b_node_array = np.zeros((len(df_ble), len(df_node)))
for i in range(len(df_ble)):
    xb = df_ble.loc[i, 'x']
    yb = df_ble.loc[i, 'y']
    zb = df_ble.loc[i, 'floor']
    b_loc = np.array([xb, yb, zb])

    for j in range(len(df_node)):
        xj = df_node.loc[j, 'x']
        yj = df_node.loc[j, 'y']
        zj = df_node.loc[j, 'floor']
        j_loc = np.array([xj, yj, zj])

        d_b_node_array[i, j] = np.linalg.norm(b_loc - j_loc)

###### 混雑度dens，maxRSSI，meanRSSIを各ビーコン各timestepごとに入れた配列（ビーコン数23*タイムステップ数90） ######
dens_jt = np.zeros((23, 60), dtype = int)
max_jt = np.zeros((23, 60), dtype = int)
mean_jt = np.zeros((23, 60), dtype = int)

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

    # numpy配列の要素はシーケンスダメらしい
    dens_jt[beacon_id-1, timestep-1] = unique_mac_count
    max_jt[beacon_id-1, timestep-1] = max_rssi
    mean_jt[beacon_id-1, timestep-1] = mean_rssi

####### リンク接続行列 ####### 向き考慮しない
def I_nod(link_data): 
    n = len(link_data)
    I = np.eye(n)
    for i in range(n):
        O = link_data.loc[i, 'O'] # 当該linkのOnode
        D = link_data.loc[i, 'D'] # Dnode
        for j in range(n):
            if ((link_data.loc[j, 'O'] == O) or (link_data.loc[j, 'O'] == D)) or (link_data.loc[j, 'D'] == O) or (link_data.loc[j, 'D'] == D): 
                I[i, j] = 1
    return(I)

####### リンク接続行列 ####### 向き考慮
def I_withd(link_data): # integratedを読み込む想定
    n = len(link_data)
    I = np.eye(n)
    for i in range(n):
        D = link_data.loc[i, 'D']
        for j in range(n):
            if link_data.loc[j, 'O'] == D: # DnodeをOnodeとするリンクを1にしている
                I[i, j] = 1
    return(I)

###### 接続行列 ###### 向き考慮しない
Ind = I_nod(df_link_odds) 

###### 接続行列 ###### 向き考慮 # ただし吸収リンクから吸収リンクへの接続は0に直す
Id = I_withd(df_link_integrated) 
Id[64, 64] = 0

staire_link = df_link_odds[df_link_odds['staire'] == 1]['linkid'].to_list() # これは必要だった

"""
###### 観測モデル内で使うためのデータ準備 ####### （これでいいのかは微妙だ．．．）
gate_node = [25, 36, # 1Fハチ公
            21, # 3F中央
            ] 

gate_link = list(df_link_odds[df_link_odds['D'].isin(gate_node)]['linkid']) # isinを覚えよう！！！
all_gate_link = df_link_odds[(df_link_odds['minami_gate'] == 1) |(df_link_odds['chuo_gate'] == 1) |(df_link_odds['hachiko_gate'] == 1)]
non_gate_link = list(all_gate_link[~all_gate_link['D'].isin(gate_node)]['linkid']) # ~をつけるとbool値が逆転

home_node = df_node[df_node['floor'] == 20.5]['nodeid'].to_list()
home_link = df_link_odds[df_link_odds['home'] == 1]['linkid'].to_list()
staire_link = df_link_odds[df_link_odds['staire'] == 1]['linkid'].to_list()

# 吸収リンクとの接続関係（流用不可）（作り替える必要）
# pre
#virtual_connect = np.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 35, 37, 43, 47, 53, 59, 61],
#                           [106, 107, 108, 108, 109, 109, 110, 111, 112, 113, 114, 114, 115, 116, 117, 103, 103, 101, 101, 102, 104, 105]])

# under
# virtual_connect = np.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 29, 33, 37, 41, 43],
#                           [78, 79, 80, 80, 81, 82, 83, 84, 85, 75, 75, 73, 73, 74, 76, 77]]) # 103, 103, 101, 101, 102, 104, 105]])

# post
virtual_connect = np.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 21, 27, 29, 31, 35, 39, 41],
                           [74, 75, 76, 76, 77, 78, 79, 80, 81, 71, 69, 69, 69, 70, 72, 73]]) # 103, 103, 101, 101, 102, 104, 105]])

"""

###### 改札の吸収リンク．homeでやるならまた別に用意しないといけない
# これはやり直し
ddata = [65] # 中央の吸収リンクのみにした
# pre: [101, 102, 103, 104, 105]
# under: [73, 74, 75, 76, 77]
# post: [69, 70, 71, 72, 73]

D = len(ddata) 


####### パラメタ初期設定 #######
xdim = 2
x0 = np.zeros(xdim)

x0[0] = -1
x0[1] = 1

####### 観測方程式 ###### # そのリンクにいる確率
def mq(max_rssi, rssi): # def mq(max_rssi, mean_rssi, dens, rssi, x):
    return (rssi/max_rssi) # 混雑を明示的に考慮できてはいないと思われる．リンク選択のところで考慮？？

####### 観測モデル ######
# とにかく回したいので欠損のないサンプルで回す．あと，15秒でtimestep刻む(30秒だと1タイムステップで複数のリンクに行けてしまうものが存在するので変になる)
def measuring_model(x):
    qall = np.zeros(len(df_link_odds)*2+2) # useridとabsorptionの列用
    for file_name in file_list:
    
            if file_name.endswith('.csv'):
                base_name = file_name.split('.')[0] # 数字部分のみ
                userid = int(base_name)
                    
                file_path = os.path.join(search_folder, file_name)
                dfi = pd.read_csv(file_path) 
                nts = dfi['timestep'].nunique()                

                # 各timestepにおける観測beaconとRSSIを取得し各timestepにおける各リンクの尤度を得る
                grouped = dfi.groupby('timestep')
                dfi_list = [group.reset_index(drop=True) for name, group in grouped]
                    
                # useridとtimestepを入れる必要はあるのか？→不明
                qi = np.zeros(len(df_link_odds)*2) # 偶奇まとめて
                for t in range(nts):
                    if t <= nts-2:
                        dfit = dfi_list[t]
                        ts = dfit.loc[0, 'timestep']

                        # tでの観測ビーコンのidとRSSIをタプルにしてリストに入れる（同時刻内の同一ビーコンのデータは，RSSIが大きい方を採用）
                        result = dfit.loc[dfit.groupby('ID')['RSSI'].idxmax()]
                        id_rssi_tuples = list(zip(result['ID'], result['RSSI']))
                        id_rssi_tuples = sorted(id_rssi_tuples, key=lambda l: l[1]) # rssiの昇順で並べ替え，上書きされるとしたら大きな値になるのでOK

                        # 各時刻の各linkの観測確率を入れる
                        qit = np.zeros(len(df_link_odds)*2)

                        # 各ビーコン付近（10m圏内）のリンクに確率を与えるのは，linklist_near_jとmqを使えばソッコーである
                        for i in range(len(id_rssi_tuples)): # 小さいリストなのでfor許してほしい
                            beaconid = id_rssi_tuples[i][0]
                            rssi = id_rssi_tuples[i][1]
                            linklist = linklist_near_j[beaconid-1] # ビーコンから10m圏内のリンクのリスト
                            max_rssi = max_jt[beaconid-1, ts-1] # tsでのbeaconのmaxのrssi
                            q = mq(max_rssi, rssi)           # 帰属確率
                                
                            for link in linklist: # linkはlinkidそのまま
                                idx = link-1    
                                #print(idx)
                                #print(qit)  
                                qit[idx] = q

                    if t == nts-1:
                        qit = np.zeros(len(df_link_odds)*2)
                        # dが21の時，最後のリンクは20→21のやつで，番号は21，idxは20
                        qit[20] = 1

                    # 時刻tのユーザiのリンク確率が入った
                    qi = np.vstack((qi, qit))
                
                ###### 最後のtimestepではd link手前のリンクに絶対行ってる
                ### +1のtimestepで吸収リンクの確率を1にする必要がある
                ### dを決めているなら，やはり最後はd（吸収リンク）を選択した，とする必要があるのでおかしい
                ### qiにnts+1列目を追加して，dに応じた箇所に確率1を割り当てる
                ### 


                # 最初の0の列を消す．qi仮完成，観測確率は付与された．続いてこれを向きを考慮して再配分する
                qi = np.delete(qi, 0, axis = 0)

                # その前に最初と最後の階段リンクへの確率を0とし，さらに各timestepで確率を正規化する（和が1になるようにならす）
                # 最初と最後のtimestepの観測結果（1次元numpyとして抽出）
                qi_first = qi[0]
                qi_last = qi[nts-1]

                # staire_linkに該当するインデックスの要素を0にする（np.arangeの方を変えた．staire_indexは1~の配列）
                qi_first[np.isin(np.arange(len(qi_first))+1, staire_link)] = 0
                qi_last[np.isin(np.arange(len(qi_last))+1, staire_link)] = 0

                # qiの各列で確率の合計が1になるように調整
                rowsum = qi.sum(axis=1)
                qi = qi / rowsum[:, np.newaxis] 

                ## 最後の観測ビーコンを参照する（最終timestepで最も観測回数の多かったやつ→データサンプリングの時，これを満たすようにフィルタリングかける）
                dfi_last = dfi_list[nts-1]
                most_common_id = dfi_last['ID'].mode()[0] # 最後のtimestepで最頻なbeaconのidが得られた        

                for j in range(len(qi)):
                    for k in range(1, len(df_link_odds)*2, 2): # df_link_integratedを参照してlink kとlink k+1のd_nodeを取得し，最終ビーコンとの距離を比較
                        klink_dnode = df_link_integrated.loc[k-1, 'D'] #Dnodeのnodeidが得られる
                        k1link_dnode = df_link_integrated.loc[k, 'D'] # kと逆方向の偶数リンクのDnodeのnodeid

                        d_k = d_b_node_array[most_common_id - 1, klink_dnode - 1]
                        d_k1 = d_b_node_array[most_common_id- 1, k1link_dnode - 1]
                        if d_k <= d_k1: 
                            continue
                        if d_k > d_k1: # ならdf[k]とdf[k+1]を入れ替える．kはlinkidで今dfはuser_idとtimestepを持つからlink kのindexはk+1(link1はindex2, ,,,)，link k+1はindexk+2
                            qi[j, k] = qi[j, k-1] #]df.iloc[j, k+2] = df.iloc[j, k+1] # k+1にkの観測確率をあげる
                            qi[j, k-1] = 0 # df.iloc[j, k+1] = 0 # kの方は0になる

                # 以上で，向きを考慮した観測確率が付与された

                # useridの列とabsorptionの列を加える
                userid_array = np.full(nts, userid)
                absorption_array = np.full(nts, most_common_id)
                qi = np.insert(qi, 0, absorption_array, axis=1)  # absorptionを先頭列に追加
                qi = np.insert(qi, 0, userid_array, axis=1)  # useridを先頭列に追加

                qall = np.vstack((qall, qi))
    qall = np.delete(qall, 0, axis = 0)
    return qall # 全タイムステップ数*userid, absorption, link数（吸収除く）の観測結果

#print(measuring_model(x0))
## とりあえずqallの動作確認をしないと→OK

###### 経路選択モデル ####### 

###### 初期設定 ######
V0 = np.full((L, D), -1) # exp(0)=1よりexp(-1)は小さくなるから?
z0 = np.exp(V0)

###### TSを設定しないといけない ###### #### 一斉に発生すると考えて良さそう→とりあえず時間構造化しないで解く
#TS = 10

V = V0
z = z0
#beta = x0[-1] # dRLなのでbeta推定．xの最後の要素をbetaとする
beta = 0.5 # dRLなのでbeta推定．xの最後の要素をbetaとする

###### 効用関数 ######
def linkv(x): 
    vinst = np.exp(df_link_integrated['length']/100 * x[0] + df_link_integrated['staire_with_esc'] * x[1]) #df_link_integrated['staire'] * x[2]) + df_link_integrated['staire_with_esc'] * x[3])
    return vinst # 出力形式はseries

###### 即時効用行列 ######
def Mset(x): 
    cost = linkv(x)
    cost = pd.concat([cost]*L, axis=1)
    cost = cost.T
    cost_numpy = cost.values # DataFrameをNumPy配列に変換
    return cost_numpy

###### 価値関数 ###### 
# inputを新しいものにしても，やはりダメなので，ここがダメなのか（そもそもの計算法を誤解しているとしか考えられない）

def newV(x):
    V = np.zeros((L, D))
    z = np.exp(V)
    
    for d in range(D): # dごとに処理
        d_linkid = ddata[d]
        
        d_index = df_link_integrated[df_link_integrated['linkid'] == d_linkid].index # ちゃんと出てる（list）
        #print(f'link{d_linkid}のindexは{d_index}')
        z[d_index, d] = 1  # 目的地がdの時，dでの期待効用Vd(d)=0より→更新後，1じゃなくなってておかしい
        M = np.zeros((L, L))
        B = np.zeros((L, 1))
        B[d_index, 0] = 1
                
        # Mをdごとに更新 # なぜ？？？？→吸収リンクへの遷移の効用を考えたいから．でもそれはいつやっても同じでは，つまりfor d loopの外でやっても同じはずなのに
        for k in range(L): # これだけみると計算結果が大きく変わりそうな気配はないな，，
            for a in range(L): 
                Ika = Id[k, a] # kもaもindexなのでd_linkidで対応．Idは普通の接続行列でいいのでは
                if Ika == 1: # 接続してなければ無関係（0のまま）
                    if a == d_index: ## 
                        M[k, a] = 1  # 吸収リンクの即時効用はexp(0)=1
                    else:
                        M[k, a] = Mset(x)[k, a]
        
        #print(f'M{M}')

        dL = 100
        zd = z[:, d].reshape(L, 1) # zのd列目のみ取り出して列ベクトル化（reshapeは保険）
        #print(f'更新前の{zd}')
        # z求解
        count = 0
        while dL >= 0.01: # 32回繰り返し計算していた
            zdd = zd.copy()
            Xz = zdd ** beta # あれここzdになってた．→直したら最後のdにも値が入った．が，それでも全て目的地dへの期待効用がほぼ同じ
            zd = np.dot(M, Xz) + B # 定義通りのはず（L*1のmatrix）
            dL = np.sum(np.abs(zdd - zd)) # np.absが各行の差分の配列．np.sumでその和をとっている．
            count += 1

        # z更新
        z[:, d] = zd[:, 0] # 収束させた後でzのd列目を更新(zdは1列しかないのでz[:, d] = zdでも良さそう)
        #print(f'更新後の{zd}')
        zd = np.where(zd == 0, 1, zd)  # 回避のための操作log(0)

        # dによってzdに変化がないとおかしいはず
        #print(f'dが{d}の時の価値関数{zd}')

        one_dim_array = np.log(zd).ravel() # 1次元配列に
        V[:, d] = one_dim_array # reshapeしなくてもOKそう

    return z

###### 対数尤度関数 ###### 
def loglikelihood(x):
    LL = 0

    # 観測モデルとの融合
    print(f'パラメタ{x}で観測モデルを回します')
    qall = measuring_model(x)
    print('観測モデルは回りました')

    # numpyのままではグルーピング不可
    mres = pd.DataFrame(qall, columns=['userid', 'absorption'] + [i for i in range(1, len(df_link_odds)*2+1)])

    # 観測結果をdで分割
    grouped = mres.groupby('absorption')
    df_list = [group for name, group in grouped]
    df_list = [data.reset_index(drop=True) for data in df_list]    

    for d in range(D):
        d_linkid = ddata[d] 
        d_index = df_link_integrated[df_link_integrated['linkid'] == d_linkid].index
        d_index = d_index.tolist()
        d_index = d_index[0] # indexはちゃんと出ている！

        # 価値関数
        z = np.exp(beta*V[:, d]).reshape(L, 1) # V所与として処理．newV内でのz（D列）とは違うので注意
        # ZD = np.array([z]*L) # コピーするには[]をつけないといけない！ZDはL*L###### 超注意！！！arrayをコピーすると3次元になってしまう．横に付け足すのはtile!!########
        ZD = np.tile(z, (1, L))
        ZD = ZD.T

        # 瞬間効用        
        M = Id * Mset(x) 
        M[:, d_index] = 1 # np.exp(0) # 吸収リンクへ遷移する際の瞬間効用は1

        # 選択確率
        Mz = (M @ z != 0) * (M @ z) + (M @ z == 0)  # MZ = np.array([Mz]*L) これ3次元になるのでnp.tileを用いる！よ
        MZ = np.tile(Mz, (1, L))  # MZ = MZ.T # MZの方は転置しない，MZから先に掛けるので．順番の都合）

        p = (M * ZD) / MZ # *で要素ごとの積（Rと同じ）
        # p = (p == 0) * 1 + (p != 0) * p  # これでp[k, a]にはk→aの遷移確率が入ってる（多分numpy）
        ## まだ早い
        #print(f'pは{p}')

        # ここでdを目的地とするユーザのデータを読み込む
        df = df_list[d]
        
        grouped2 = df.groupby('userid')
        df_list2 = [group for name, group in grouped2] # df_list2の長さが目的地をdとするuserの数に等しい
        df_list2 = [data2.reset_index(drop=True) for data2 in df_list2]

        Ld = 0 # dのための対数尤度
        #print(len(df_list2))
        count = 0

        for i in range(len(df_list2)): #iは個人
            df_indivi = df_list2[i] # 各自のデータ # 列カラムはuser_id, timestep, link1~117（columnの観測確率，absoptionの120列
            userid = df_indivi.loc[0, 'userid']
            #print(f'今userid{userid}のサンプルです')

            data_subset = df_indivi.iloc[:, 2:len(df_link_odds)*2+2]
            #print(data_subset) # 良さそう
            qi = data_subset.to_numpy() # q[t, k]が，個人のtimestep t番目のlink k+1の観測確率に対応
            
            li = 1 # 個人ごとの初期の尤度

            # timestepごとにステップ毎に尤度出して行く
            for t in range(1, len(qi)): 
            #そもそもtimestep1は前を参照しようがないのでtimestep2から処理
                # 時刻tにおいてq != 0のlinkidを取得→measured_link
                measured_links = np.where(qi[t] != 0)[0] # list形式．index
                measured_links = measured_links + 1 # linkid

                # 時刻tの時の観測尤度の初期化
                lt = 0

                # 時刻tでの各観測リンクに対しての操作
                for measured_link in measured_links: # measured_linkはlinkid
                    l_for_each = 0
                    measured_column = Id[:, measured_link-1] # linkid-1なのでlink index．measured_linkをdとするリンクのところが1になってる
                    pre_link_list_index = np.where(measured_column == 1)[0] # measured_linkidに接続するリンクpre_link集合を得る．接続行列Idを参照
                    pre_link_list = pre_link_list_index + 1 # linkid
                    
                    #print(f'今のlinkidは{measured_link}で，接続してるはずのlinkのlinkidは{pre_link_list}') # 自分は自分に接続してるので，
                    
                    for pre_link in pre_link_list:
                        #print(f'prelink{pre_link}, prelinklist{pre_link_list}')
                        l_for_each = l_for_each + qi[t-1, pre_link-1] * p[pre_link-1, measured_link-1] * qi[t, measured_link-1]
                        
                        #print(q[t-1, pre_link-1], p[pre_link-1, measured_link-1], q[t, measured_link-1])
                        #print(f'{l_for_each})
                    
                    lt = lt + l_for_each # 各measured_linkのlink尤度を積み上げる
                    # これで時刻tの尤度ltがでた．後はLiにltを順次かけていく（対数尤度なら，足していく）
                
                #print(f'時刻{t}で尤度ltは{lt}') # 時刻1でltが0になっている→残り全部0になってしまう，ltが0ならliが0になってlliが0になってしまう．
                li = li * lt
                #print(f'時刻{t}で尤度liは{li}')
                        
            # これで個人iの尤度liがでた
            li = (li == 0) + (li != 0) * li
            Ld += math.log(li)
            count += 1

        LL += Ld
        print(f"link{d}の操作終わり，計算したuserは{count}人で，今尤度は{LL}です")

    print('以上で一回loglikelihoodが回りました')
    return -LL

###### 推定部分 ###### 
dL = 100
n = 0

x_init = np.zeros(2)
#bounds = [(0, None), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 1)]

while dL >= 1:
    print(f"<<<<<<<<<<<<<<<<now at {n}th loop in while cond>>>>>>>>>>>>>>>>>>")
    n += 1
    x = x0
    V = V0
    z = z0 # 価値関数固定

    # 構造推定part1：対数尤度関数最大化
    res = minimize(loglikelihood, x, method='BFGS') #, options={"maxiter":10,"return_all":True}) 
    
    # 推定値でパラメタxを更新
    x0 = res.x

    print("x0  =", x0)
    print("lnL =", res.fun)

    # 構造推定part2：価値関数更新
    z0 = newV(x0) # 119*2のnumpy配列
    zz0 = (z0 == 0) * 1 + (z0 != 0) * z0 # pの算出と構造同じなのでOKのはず
    V0 = np.log(zz0)

    # 収束条件の計算（パラメタが収束しているか）
    dL = np.sum(np.abs(x - x0)) # 収束していればxでもx0でも同じ
    print("dL =", dL)
    print(n)

hhh = res.hess_inv #.todense()
print(n)
tval = x0 / np.sqrt(np.diag(hhh))
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
print("説明変数の表示")
print(linkv)
print("NFXP")
print("初期尤度 = ", L0)
print("最終尤度 = ", LL)
print("ρ値 = ", (L0 - LL) / L0)
print("修正済ρ値 = ", (L0 - (LL - len(x))) / L0)
print("パラメータ初期値 = ", x_init)
print("パラメータ推定値 = ", x0)
print("時間割引率 = ", beta)
print("t値 = ", tval)