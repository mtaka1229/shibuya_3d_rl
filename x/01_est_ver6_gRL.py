# 観測方程式の考え方が全くわかってなかったので一からやり直す
# nodeの設定もやり直す
# 時間構造化したgRLでコード書き直す（後ろ向きで求めるので逆行列or繰り返し計算なし）

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
df_quater = pd.read_csv('/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20230130_08/20230130_0815.csv')

df_link_odds = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_corrected/shibuya_link_post_odds.csv")
l = len(df_link_odds)

df_link_integrated = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_corrected/shibuya_link_post_integrated.csv')
L = len(df_link_integrated)

search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20230130_08/quater_address750"
file_list = list(os.listdir(search_folder))

df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node.csv')
df_ble = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/ble_nw.csv')

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

        d_node_array[i, j] = np.linalg.norm(b_loc - j_loc)

###### 混雑度dens，maxRSSI，meanRSSIを各ビーコン各timestepごとに入れた配列（ビーコン数23*タイムステップ数90） ######
dens_jt = np.zeros((23, 90), dtype = int)
max_jt = np.zeros((23, 90), dtype = int)
mean_jt = np.zeros((23, 90), dtype = int)

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

###### 接続行列 ###### 向き考慮
Id = I_withd(df_link_integrated) 

###### 観測モデル内で使うためのデータ準備 ####### （これでいいのかは微妙だ．．．）
gate_node = [33, 36, # 1Fハチ公
            21, # 3F中央
            26, 27] # 1F南

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


####### パラメタ初期設定 #######
xdim = 5
x0 = np.zeros(xdim)

####### 観測方程式 ###### # そのリンクにいる確率
def mq(max_rssi, rssi, x): # def mq(max_rssi, mean_rssi, dens, rssi, x):
    return (rssi/max_rssi*(1+x[0])) # 混雑を明示的に考慮できてはいないと思われる．リンク選択のところで考慮？？

####### 観測モデル ######
# とにかく回したいので欠損のないサンプルで回す．あと，15秒でtimestep刻む(30秒だと1タイムステップで複数のリンクに行けてしまうものが存在するので変になる)
def measuring_model(x):
    qlist = []
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
                    dfit = dfi_list[t]
                    ts = dfit.loc[0, 'timestep']

                    # tでの観測ビーコンのidとRSSIをタプルにしてリストに入れる（同時刻内の同一ビーコンのデータは，RSSIが大きい方を採用）
                    result = dfit.loc[dfit.groupby('ID')['RSSI'].idxmax()]
                    id_rssi_tuples = list(zip(result['ID'], result['RSSI']))
                    id_rssi_tuples = sorted(id_rssi_tuples, key=lambda l: l[1]) # rssiの昇順で並べ替え，上書きされるとしたら大きな値になるのでOK

                    # 各時刻の各linkの観測確率を入れる
                    qit = np.zeros(len(df_link_odds*2))

                    # 各ビーコン付近（10m圏内）のリンクに確率を与えるのは，linklist_near_jとmqを使えばソッコーである
                    for i in range(len(id_rssi_tuples)): # 小さいリストなのでfor許してほしい
                        beaconid = id_rssi_tuples[i][0]
                        rssi = id_rssi_tuples[i][1]
                        linklist = linklist_near_j[beaconid-1] # ビーコンから10m圏内のリンクのリスト
                        max_rssi = max_jt[beaconid-1, ts-1] # tsでのbeaconのmaxのrssi
                        q = mq(max_rssi, rssi, x)           # 帰属確率
                            
                        for link in linklist: # linkはlinkidそのまま
                            idx = link-1      
                            qit[idx] = q

                    # 時刻tのユーザiのリンク確率が入った
                    qi = np.vstack((qi, qit))

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

                ## 最後の観測ビーコンを参照する（最終timestepで最も観測回数の多かったやつ）
                dfi_last = dfi_list[nts-1]
                most_common_id = dfi_last['ID'].mode()[0] # 最後のtimestepで最頻なbeaconのidが得られた

                ######## これを吸収位置の判定に使うか？？？→列追加する方がいいんか？

                for j in range(len(qi)):
                    for k in range(1, len(df_link_odds)*2, 2): # df_link_integratedを参照してlink kとlink k+1のd_nodeを取得し，o_nodeとの距離を比較
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
                # 各ユーザの経路確率の結果をqlistに追加
                qlist.append(qi)

    return qlist


###### 経路選択モデル ####### 

###### 初期設定 ######

# 改札の吸収リンク．homeでやるならまた別に用意しないといけない
# これはやり直し
ddata = [69, 70, 71, 72, 73] 
# pre: [101, 102, 103, 104, 105]
# under: [73, 74, 75, 76, 77]
# post: [69, 70, 71, 72, 73]

D = len(ddata) 
V0 = np.full((L, D), -1) # exp(0)=1よりexp(-1)は小さくなるから?
z0 = np.exp(V0)

###### TSを設定しないといけない ###### #### 一斉に発生すると考えて良さそう
TS = 10

V = V0
z = z0
beta = x0[-1] # dRLなのでbeta推定．xの最後の要素をbetaとする
#beta = 0.5 # dRLなのでbeta推定．xの最後の要素をbetaとする

###### 時空間プリズムIlistを用意する必要 ######
Ilist 

###### 効用関数 ###### 
def linkv(x):
    vinst = np.exp(df_link_integrated['length']/10 * x[1] + df_link_integrated['staire'] * x[2] + df_link_integrated['staire_with_esc'] * x[3])
    return vinst


###### 価値関数 ######
def newV(x):
    z = np.ones((TS + 1, L, D))  # exp(Vd)
    vinst = linkv(x)

    for d in range(D):
        M = np.zeros((TS, L, L)) # Mは各timestepごとにある

        # これは指示変数delta
        Id = Ilist[d]  # d番目のIlistをIdに入れている. Ilistは時空間プリズム=時空間的接続行列. Id[ts,k,a] <- 1,

        for ts in range(TS): # 即時効用行列（各時刻に対してIdが決まってるので，Mも各時刻に対して決まることになる）
            Mts = Id[ts] * vinst
            M[ts] = Mts
        
        for tt in range(TS - 1, -1, -1):
            zi = np.dot(M[tt], z[tt + 1, :, d] ** beta)
            z[tt, :, d] = np.where(zi == 0, 1, zi)

    return z

###### 対数尤度関数 ######
def loglikelihood(x):
    LL = 0
    z = newV(x)
    V = np.log(z)
    vinst = linkv(x)


    # 観測モデルの参照
    print(f'パラメタ{x}で観測モデルを回します')
    m_res = measuring_model(x)
    print('観測モデルは回りました')



    for d in range(D):
        z = np.exp(beta*V[2, :, d]).reshape(L, 1) # なんで2番目なんだ
        ZD = np.tile(z, (1, L))
        ZD = ZD.T 

        M = Id[1] * vinst # なんでId[1]なのかは不明．そもそもIdがわかってない
        ##### 吸収リンクのところは1にしないのか？？
        
        # 選択確率
        Mz = (M @ z != 0) * (M @ z) + (M @ z == 0)  # MZ = np.array([Mz]*L) これ3次元になるのでnp.tileを用いる！よ
        MZ = np.tile(Mz, (1, L)) 

        # 