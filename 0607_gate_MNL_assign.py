
import pandas as pd 
import numpy as np 
import os 
import math
import heapq
import random
from scipy.optimize import minimize

base_path = '/Users/takahiromatsunaga/res2023/PPcameraTG/PP'
hour, signed_point_percent = 10, 0 # 10時代の結果，提示率は40%
df_mezzo_link = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/map/shibuya_mezzo_link.csv'))
df_mezzo_node = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/map/shibuya_mezzo_node.csv'))
df_mezzo_nw = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/map/shibuya_mezzo_nodebased_network.csv'))
df_gate = pd.read_csv(os.path.join(base_path, 'PP_webq/gate_MNL/gate.csv'))
df_destination = pd.read_csv(os.path.join(base_path, 'PP_webq/gate_MNL/destination.csv'))
df_micro_node = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/shibuya_stanw/micro_node_4326.csv'))

df_demand = pd.read_csv(os.path.join(base_path, f'PP_webq/destination_assignment/dest_assign_res/destcar_assign_{hour}_{signed_point_percent}.csv'))
# /Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_webq/destination_assignment/dest_assign_res/destcar_assign_10_60.csv
### 時間帯とポイント提示率を指定（パイパラ）
# macroの結果．
df_macro_res = pd.read_csv(os.path.join(base_path, f'../../results/macro_result/macro_result0606/point_time_{signed_point_percent}.csv' ))
# 指定時間に渋谷に来る人の数
arrival = df_macro_res[df_macro_res['time'] == hour]['shibuya_arrive'].iloc[0]

NG = len(df_gate)
ND = len(df_destination)
OD = len(df_demand)

hachi_loc = np.array([df_gate.loc[0, 'lat'], df_gate.loc[0, 'lon']])
chuo_loc = np.array([df_gate.loc[1, 'lat'], df_gate.loc[1, 'lon']])
minami_loc = np.array([df_gate.loc[2, 'lat'], df_gate.loc[2, 'lon']])
newminami_loc = np.array([df_gate.loc[3, 'lat'], df_gate.loc[3, 'lon']])

hachi_floor = df_gate.loc[0, 'floor']
chuo_floor = df_gate.loc[1, 'floor']
minami_floor = df_gate.loc[2, 'floor']
newminami_floor = df_gate.loc[3, 'floor']

nodeid_mezzo_list = sorted(df_mezzo_node['id'].unique()) 
linkid_mezzo_list = sorted(df_mezzo_link['id'].unique())

N_mezzo = len(df_mezzo_node)
L_mezzo = len(df_mezzo_link)

length_mat_mezzo = np.zeros((N_mezzo, N_mezzo)) 

### ture parameters ## 有意じゃなかったものは0にしてる（健全）
x_res = np.array([-1.62, -1.03, 0.74, 0.52, 1.23, 0, 0])
### ポイント設定
point_gate = 3 # 1~4
point_amount = 0 # 0の場合，どこにもポイントをあげない＝無政策時の結果

# リンク間距離行列
for i in range(len(df_mezzo_nw)):
    kn = int(df_mezzo_nw.loc[i, 'k']) # これはid
    an = int(df_mezzo_nw.loc[i, 'a'])
    k = nodeid_mezzo_list.index(kn) # node knのindex
    a = nodeid_mezzo_list.index(an)
    kan = None ## 以降でif文がpassされる時，kanは未定義のまま参照されることになる．これだとダメなので最初に定義しておく．
    if not df_mezzo_link[(df_mezzo_link['onode'] == kn) & (df_mezzo_link['dnode'] == an)].empty:
        kan = df_mezzo_link[(df_mezzo_link['onode'] == kn) & (df_mezzo_link['dnode'] == an)]['id'].iloc[0] # リンクデータは無向リンクで考えているので双方向を考える
    if not df_mezzo_link[(df_mezzo_link['onode'] == an) & (df_mezzo_link['dnode'] == kn)].empty:
        kan = df_mezzo_link[(df_mezzo_link['onode'] == an) & (df_mezzo_link['dnode'] == kn)]['id'].iloc[0] # 
    if not kan == None:
        ka = linkid_mezzo_list.index(kan) # linkidがkanのリンクのindex
        length_mat_mezzo[k, a] = (df_mezzo_link.loc[ka, 'length']) # kaはindexなのでこのままでOK

def Dis(lon_error, lat_error):
    R = 6378137
    THETA = 33.85/180 * math.pi
    lon_1 = 2*math.pi * R * math.cos(THETA) / 360 # 経度1度あたりの距離を計算しておく
    lat_1 = 2 * math.pi * R / 360 # 緯度1度あたりの距離を計算しておく
    # lon_error, lat_errorにそれぞれlon_1, lat_1を掛けて各水平距離を計算
    lon_dis = lon_error * lon_1
    lat_dis = lat_error * lat_1
    dis = lon_dis ** 2 + lat_dis ** 2
    return dis ** 0.5

def dijkstra(adj, orin, nodeid_list, N): # adjはノード間distを格納してる行列!!．lenght_matを使えばいい． # node id をinputにする
    # N = self.n_node # Nはノード数．すでにグローバル変数として与えている
    ori = nodeid_list.index(orin) # indexに変換
    min_cost = [float('inf')] * N
    checked = [False] * N  # 確定したか否か
    forward = [None] * N  # 最短経路をたどるための直前ポインタ
    q = []  # 優先度付きキュー，各要素はタプル(コスト，ノードID)，未確定ノードのうちコストが有限のもの
    min_cost[ori] = 0 # oriはindexなのでそのまま
    forward[ori] = orin # これはnodeidを入れる（indexではない）
    heapq.heappush(q, (0, orin))
    while q:
        _, vn = heapq.heappop(q)  # 費用が最小のノードをvとする  # _はタプルの最初の要素＝コスト# vnはnode id
        v = nodeid_list.index(vn)  # vはindex
        if checked[v]:
            continue
        checked[v] = True  # ノードvを確定
        for i in range(N):
            dist = adj[v, i]  # ノードvとiの間の距離
            if not checked[i] and dist != 0:  # 未確定隣接ノードについて
                tmp_cost = min_cost[v] + dist
                if tmp_cost < min_cost[i]:
                    # コストとポインタを更新
                    min_cost[i] = tmp_cost
                    forward[i] = vn  # id
                    inn = nodeid_list[i]
                    heapq.heappush(q, (min_cost[i], inn))
    return forward, min_cost # min_costが各nodeからoriまでの最短経路．これをdesに対しても拡張すれば良さそう

_, mincost_from_hachi = dijkstra(length_mat_mezzo, 743, nodeid_mezzo_list, N_mezzo) 
_, mincost_from_chuo = dijkstra(length_mat_mezzo, 750, nodeid_mezzo_list, N_mezzo) 
_, mincost_from_minami = dijkstra(length_mat_mezzo, 745, nodeid_mezzo_list, N_mezzo) 
_, mincost_from_newminami = dijkstra(length_mat_mezzo,560, nodeid_mezzo_list, N_mezzo) 

def find_closest_node(lat, lon, df_node): 
    distances = []
    for index, row in df_node.iterrows():
        node_lat = row['lat'] ### df_nodeのlat, lonが逆になってたのでやり直し
        node_lon = row['lon']
        distance = Dis(node_lon - lon, node_lat - lat)
        distances.append(distance)
    closest_node_index = np.argmin(distances)
    return df_node.loc[closest_node_index, 'id']

### carnodeとgateは直線距離で固定なのでmatrix用意しておいて後から参照するのがいい
carnode_list = [i for i in range(1, 16)] # 1~15の整数 # 工事中後のホーム上ノード番号
gate_list = df_gate['id'] # 1がhachi, 2がchuo, 3がminami, 4がnewminami
gate_car_distmat = np.zeros((len(gate_list), len(carnode_list)))
for i in range(len(gate_list)):
    gate_loc = np.array([df_gate.loc[i, 'lat'], df_gate.loc[i, 'lon']])
    for j in range(len(carnode_list)):
        carnode = carnode_list[j] # 降車位置のnodeid
        carnode_lat = df_micro_node[df_micro_node['id'] == carnode]['lat'].iloc[0]
        carnode_lon = df_micro_node[df_micro_node['id'] == carnode]['lon'].iloc[0]
        carnode_loc = np.array([carnode_lat, carnode_lon])
        lat_error, lon_error = abs(gate_loc - carnode_loc)
        gate_car_distmat[i, j] = Dis(lon_error, lat_error)

### 確認 
print('gate_car_distmat', gate_car_distmat)


def assign(x):
    x_mezzo_length, x_micro_length, x_level, x_point, x_hachiko, x_chuo, x_minami = x
    df_res = pd.DataFrame(columns=['dcarnode', 'dgate']) # demandで束ねたいけど無理か？
    
    ### 号車，目的地の組で10*8 = 80ODセットで回す？？ 
    #で一旦分けるとかした方がいいか？？
    res_dict = {}
    for od in range(OD):
        res = [] # 降車位置と選択された改札のidのタプルを返す
        #oddemand = df_demand.loc[od, 'demands'] 
        dcar_nodeid = df_demand.loc[od, 'car']
        dest = df_demand.loc[od, 'd'] ## 需要データの最終目的地

        if dest == 7:
            d_lat = df_demand.loc[od, 'lat']
            d_lon = df_demand.loc[od, 'lon']
            # d_floor = df_diary.loc[i, '']
            # d_loc = np.array([d_lat, d_lon])
            # 最寄りのノードを持ってくる
            d_mezzo_nodeid = find_closest_node(d_lat, d_lon, df_mezzo_node)
            d_mezzonode_idx = nodeid_mezzo_list.index(d_mezzo_nodeid)
            d_floor = 1

        else:
            # 番号に応じて決まる距離をあらかじめmatrixで用意しておく
            d_floor = df_destination[df_destination['id'] == dest]['floor'].iloc[0]
            d_mezzo_nodeid = df_destination[df_destination['id'] == dest]['mezzo_nodeid'].iloc[0]
            d_mezzonode_idx = nodeid_mezzo_list.index(d_mezzo_nodeid)
        
        dist_dest_hachi = mincost_from_hachi[d_mezzonode_idx]
        dist_dest_chuo = mincost_from_chuo[d_mezzonode_idx]
        dist_dest_minami = mincost_from_minami[d_mezzonode_idx]
        dist_dest_newminami = mincost_from_newminami[d_mezzonode_idx]

        ## micro_length
        #### gate_car_distmatを参照
        dcar_nodeidx = carnode_list.index(dcar_nodeid) # 実際にはdcar_nodeid-1と同じ
        dist_car_hachi = gate_car_distmat[0, dcar_nodeidx]
        dist_car_chuo = gate_car_distmat[1, dcar_nodeidx]
        dist_car_minami = gate_car_distmat[2, dcar_nodeidx]
        dist_car_newminami = gate_car_distmat[3, dcar_nodeidx]

        ## level_dum
        level_hachi_dum = 1 if hachi_floor == d_floor else 0
        level_chuo_dum = 1 if chuo_floor == d_floor else 0
        level_minami_dum = 1 if minami_floor == d_floor else 0
        level_newminami_dum = 1 if newminami_floor == d_floor else 0

        ### 各改札の効用を計算（odごとで定義）
        u_hachiko = np.exp((dist_dest_hachi/1000) * x_mezzo_length + (dist_car_hachi/100) * x_micro_length + x_hachiko  + level_hachi_dum * x_level) 
        u_chuo = np.exp((dist_dest_chuo/1000) * x_mezzo_length + (dist_car_chuo/100) * x_micro_length + x_chuo  + level_chuo_dum * x_level)
        u_minami = np.exp((dist_dest_minami/1000) * x_mezzo_length + (dist_car_minami/100) * x_micro_length + x_minami + level_minami_dum * x_level)
        u_newminami = np.exp((dist_dest_newminami/1000) * x_mezzo_length + (dist_car_newminami/100) * x_micro_length + level_newminami_dum * x_level)
        
        if point_gate == 1: # ハチ公
            u_hachiko *= np.exp(point_amount/10 * x_point)
        elif point_gate == 2: # 中央
            u_chuo *= np.exp(point_amount/10 * x_point)
        elif point_gate == 3: # 南口
            u_minami *= np.exp(point_amount/10 * x_point)
        elif point_gate == 4: # 新南口 ## 新南口にめちゃポイント出してどうなるか，など
            u_newminami *= np.exp(point_amount/10 * x_point)

        deno = u_hachiko + u_chuo + u_minami + u_newminami

        P_hachiko = u_hachiko / deno
        P_chuo = u_chuo / deno
        P_minami = u_minami / deno
        P_newminami = u_newminami / deno ## ここまでで選択確率出た

        ## 累積確率列
        P_accum = np.zeros(len(df_gate)) 
        P_accum[0] = P_hachiko
        P_accum[1] = P_accum[0] + P_chuo
        P_accum[2] = P_accum[1] + P_minami
        P_accum[3] = P_accum[2] + P_newminami

        ### 個人について確率的に改札を配分
        #for i in range(oddemand):
        rand = np.random.rand() # 各改札の選択を乱数で決める
        selected_gate = 0
        #### 適当に割り当てる
        if rand <= P_accum[0]:
            selected_gate = 1
        else:
            for j in range(1, len(P_accum)): # 1, 2, 3
                if P_accum[j-1] <= rand <= P_accum[j]:
                    selected_gate = j+1 # df_gateにおけるgateidを入れる．それはj+1
                else:
                    continue
        new_row = {'dcarnode': dcar_nodeid, 'dgate': selected_gate}
        # i_od = (dcar_nodeid, selected_gate) # タプルにする ### selected_gateから出口を配分したいが
        # print(i_od)
        # res.append(i_od)
        df_res = df_res.append(new_row, ignore_index=True)


    # resのうち，重複しているものはその数をカウントする

    # for item in res:
    #     if item in res_dict.keys():
    #         res_dict[item] += 1
    #     else:
    #         res_dict[item] = 1
    df_res_counts = df_res.groupby(['dcarnode', 'dgate']).size().reset_index(name='counts')
    return df_res_counts


#### まあほとんどできたのでO分布与えるのとミクロシミュレーションの方に労力シフト

df_res_counts = assign(x_res)

print(df_res_counts)
#df_res_counts.to_csv(f'/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_webq/gate_MNL/gate_assignment_0610/gate_assign_gate{point_gate}_{point_amount}point_{signed_point_percent}percent.csv')


# 'abs'列を追加するための処理
new_rows = []

for _, row in df_res_counts.iterrows():
    dcarnode = row['dcarnode']
    dgate = row['dgate']
    counts = row['counts']
    
    if dgate == 1:
        count1 = counts // 2
        count2 = counts - count1
        new_rows.append({'dcarnode': dcarnode, 'dgate': dgate, 'counts': count1, 'abs': 67}) #ハチ公
        new_rows.append({'dcarnode': dcarnode, 'dgate': dgate, 'counts': count2, 'abs': 68})
    elif dgate == 2:
        count1 = counts // 2
        count2 = counts - count1
        new_rows.append({'dcarnode': dcarnode, 'dgate': dgate, 'counts': count1, 'abs': 69}) # minami
        new_rows.append({'dcarnode': dcarnode, 'dgate': dgate, 'counts': count2, 'abs': 70})
    elif dgate == 3:
        new_rows.append({'dcarnode': dcarnode, 'dgate': dgate, 'counts': counts, 'abs': 66}) # chuo
    elif dgate == 4:
        new_rows.append({'dcarnode': dcarnode, 'dgate': dgate, 'counts': counts, 'abs': 71}) # 新南

df_res_counts_abs = pd.DataFrame(new_rows)
df_res_counts_abs.to_csv(f'/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_webq/gate_MNL/gate_assignment_0610/gate_assign_gate{point_gate}_{point_amount}point_{signed_point_percent}percent_abs.csv')