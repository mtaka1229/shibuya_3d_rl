# gate choice MNL 2024/06/04
# input:diary data

import pandas as pd 
import numpy as np 
import os 
import math
import heapq
from scipy.optimize import minimize

base_path = '/Users/takahiromatsunaga/res2023'
# 最も重要なinputデータ, 0601webdiary.ipynbより出力
# df_diary = pd.read_csv(os.path.join(base_path, 'PPcameraTG/PP/PP_webq/q_ans/diary_cleaned_dcar_latlon_point6.csv')) # pointのところがデータ間違ってた
df_diary = pd.read_csv(os.path.join(base_path, 'PPcameraTG/PP/PP_webq/q_ans/diary_cleaned_odcar_odlatlon_point.csv')) # pointのところがデータ間違ってた
df_mezzo_link = pd.read_csv(os.path.join(base_path, 'shibuya_nw/map/shibuya_mezzo_link.csv'))
df_mezzo_node = pd.read_csv(os.path.join(base_path, 'shibuya_nw/map/shibuya_mezzo_node.csv'))
df_mezzo_nw = pd.read_csv(os.path.join(base_path, 'shibuya_nw/map/shibuya_mezzo_nodebased_network.csv'))
df_micro_node = pd.read_csv(os.path.join(base_path, 'shibuya_nw/shibuya_stanw/micro_node_4326.csv'))
df_gate = pd.read_csv(os.path.join(base_path, 'PPcameraTG/PP/PP_webq/gate_MNL/gate.csv'))
df_destination = pd.read_csv(os.path.join(base_path, 'PPcameraTG/PP/PP_webq/gate_MNL/destination.csv'))

NG = len(df_gate)
ND = len(df_destination)
DIARY = len(df_diary)

odindex = 'o' # 渋谷を出るか，渋谷に来るか．出るとo，来る時はd

# 変化しない情報
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

x_dim = 6 # パラメータ数

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

def find_closest_node(lat, lon, df_node):
    distances = []
    for index, row in df_node.iterrows():
        node_lat = row['lat'] ### df_nodeのlat, lonが逆になってたのでやり直し
        node_lon = row['lon']
        distance = Dis(node_lon - lon, node_lat - lat)
        distances.append(distance)
    closest_node_index = np.argmin(distances)
    return df_node.loc[closest_node_index, 'id']

# length matrix of gate location and destination(node) # as of parameter, it may be better to logalize the length term
# gate_d_len_mat = np.zeros((NG, ND)) # gateから各目的地への距離を入れる
# for i in range(NG):
#     gate_lat, gate_lon = df_gate.loc[i, 'lat'], df_gate.loc[i, 'lon']
#     gate_loc = np.array([gate_lat, gate_lon])
#     for j in range(ND):
#         d_lat, d_lon = df_destination.loc[j, 'lat'], df_destination.loc[j, 'lon']
#         d_loc = np.array([d_lat, d_lon])
#         lat_error, lon_error = abs(gate_loc - d_loc)
#         dis_ij = Dis(lon_error, lat_error) # 距離（m）
#         gate_d_len_mat[i, j] = dis_ij

# gate_d_level_mat = np.zeros((NG, ND))
# for i in range(NG):
#     gate_level = int(df_gate.loc[i, 'floor'])
#     for j in range(ND):
#         d_level = int(df_destination.loc[j, 'level'])
#         if gate_level == d_level:
#             gate_d_level_mat[i, j] = 1

# 個人ごとに説明変数を記録
diary_list = []
count = 0
for i in range(DIARY):
    userid = df_diary.loc[i, 'id']
    day = df_diary.loc[i, 'day']
    ival = [] # valueを入れていくリスト

    ## d_gate
    # gate_id = df_diary.loc[i, 'd_gate']  ### 到着時の改札なのでd
    # gate_rows = df_gate[df_gate['id'] == gate_id]
    # if not gate_rows.empty:
    #     gate_mezzo_nodeid = gate_rows['mezzo_nodeid'].iloc[0]
    #     # gate_mezzo_nodeid = df_gate[df_gate['id'] == gate_id]['mezzo_nodeid'].iloc[0]
    #     gate_rows2 = df_mezzo_node[df_mezzo_node['id'] == gate_mezzo_nodeid]
    #     if not gate_rows2.empty:
    #         gate_lat = gate_rows2['lat'].iloc[0]
    #         gate_lon = gate_rows2['lon'].iloc[0]
    #         # gate_floor = df_gate[df_gate['id'] == gate_id]['floor'].iloc[0]
    #         gate_floor = gate_rows['floor'].iloc[0]
    #         gate_loc = np.array([gate_lat, gate_lon]) # gateの緯度経度でた
    # else:
    #     # print(f"Error: gate_id {gate_id} not found in df_gate")
    #     diary_list.append([])
    #     continue  # または、適切なエラーハンドリングを行う

    # car_node
    if odindex == 'o':
        # car_node = df_diary.loc[i, 'dcar_node'] # micro nwでのnodeid #### 到着時の利用車両番号
        car_node = df_diary.loc[i, 'ocar_node'] # micro nwでのnodeid #### 到着時の利用車両番号
        car_rows = df_micro_node[df_micro_node['id'] == car_node]
    else:
        car_node = df_diary.loc[i, 'dcar_node'] # micro nwでのnodeid #### 到着時の利用車両番号
        car_rows = df_micro_node[df_micro_node['id'] == car_node]
    
    if not car_rows.empty:
        car_node_lat = car_rows['lat'].iloc[0] # 平面直角
        car_node_lon = car_rows['lon'].iloc[0]
        ## 平面直角から座標系への変換 # 結局使わず，car_nodeを4326でQGIS上で打ち直した
        #source_coords = (car_node_x, car_node_y)
        #latlon = convert_to_latlon(source_coords)
        #car_node_lat = latlon[0]
        #car_node_lon = latlon[1]
        car_loc = np.array([car_node_lat, car_node_lon])

        lat_error_car_hachi, lon_error_car_hachi = abs(car_loc - hachi_loc)
        lat_error_car_chuo, lon_error_car_chuo = abs(car_loc - chuo_loc)
        lat_error_car_minami, lon_error_car_minami = abs(car_loc - minami_loc)
        lat_error_car_newminami, lon_error_car_newminami = abs(car_loc - newminami_loc)

        dist_car_hachi = Dis(lon_error_car_hachi, lat_error_car_hachi) # でた
        dist_car_chuo = Dis(lon_error_car_chuo, lat_error_car_chuo) # でた
        dist_car_minami = Dis(lon_error_car_minami, lat_error_car_minami) # でた
        dist_car_newminami = Dis(lon_error_car_newminami, lat_error_car_newminami) # でた

    else:
        # print(f"Error: car_node {car_node} not found in df_micro_node")
        diary_list.append([])
        continue  # または、適切なエラーハンドリングを行う

    # 最後にdestination
    dest_id = df_diary.loc[i, 'd'] if odindex == 'd' else df_diary.loc[i, 'o']

    if dest_id == 0: ### 目的地が1-7の候補地以外の場合，各トラジェクトリの最終時点での位置情報を目的地とする
        d_lat = df_diary.loc[i, 'd_lat'] if odindex == 'd' else df_diary.loc[i, 'o_lat']
        d_lon = df_diary.loc[i, 'd_lon'] if odindex == 'o' else df_diary.loc[i, 'o_lon']
        if d_lat == 0 or d_lon == 0:
            diary_list.append([])
            continue
        else:
            d_loc = np.array([d_lat, d_lon])
            # 最寄りのノードを持ってくる
            d_mezzo_nodeid = find_closest_node(d_lat, d_lon, df_mezzo_node)
            d_floor = 1

    else: # o or dの番号が0でない場合
        d_mezzo_nodeid = df_destination[df_destination['id'] == dest_id]['mezzo_nodeid'].iloc[0] # これでいいんじゃね
        #d_lat = df_mezzo_node[df_mezzo_node['id'] == d_nodeid]['lat'].iloc[0]
        #d_lon = df_mezzo_node[df_mezzo_node['id'] == d_nodeid]['lon'].iloc[0]
        d_floor = df_destination[df_destination['id'] == dest_id]['floor'].iloc[0]
        #d_loc = np.array([d_lat, d_lon])

    level_hachi_dum = 1 if hachi_floor == d_floor else 0
    level_chuo_dum = 1 if chuo_floor == d_floor else 0
    level_minami_dum = 1 if minami_floor == d_floor else 0
    level_newminami_dum = 1 if newminami_floor == d_floor else 0

    _, mincost_from_hachi = dijkstra(length_mat_mezzo, 743, nodeid_mezzo_list, N_mezzo) 
    _, mincost_from_chuo = dijkstra(length_mat_mezzo, 750, nodeid_mezzo_list, N_mezzo) 
    _, mincost_from_minami = dijkstra(length_mat_mezzo, 745, nodeid_mezzo_list, N_mezzo) 
    _, mincost_from_newminami = dijkstra(length_mat_mezzo,560, nodeid_mezzo_list, N_mezzo) 

    d_mezzonode_idx = nodeid_mezzo_list.index(d_mezzo_nodeid)
    dist_dest_hachi = mincost_from_hachi[d_mezzonode_idx]
    dist_dest_chuo = mincost_from_chuo[d_mezzonode_idx]
    dist_dest_minami = mincost_from_minami[d_mezzonode_idx]
    dist_dest_newminami = mincost_from_newminami[d_mezzonode_idx]

    point_gate = df_diary.loc[i, 'point_gate'] # 1:ハチ公, 2:中央, 3:南
    point_amount = df_diary.loc[i, 'point_amount'] # 20, 40, 60

    print(f'userid:{userid}, day{day}, diary結果は{[dist_car_hachi, dist_car_chuo, dist_car_minami, dist_car_newminami, dist_dest_hachi, dist_dest_chuo, dist_dest_minami, dist_dest_newminami, level_hachi_dum, level_chuo_dum, level_minami_dum, level_newminami_dum, point_gate, point_amount]}')

    diary_list.append(
        [dist_car_hachi, dist_car_chuo, dist_car_minami, dist_car_newminami, 
         dist_dest_hachi, dist_dest_chuo, dist_dest_minami, dist_dest_newminami, 
         level_hachi_dum, level_chuo_dum, level_minami_dum, level_newminami_dum,
         point_gate, point_amount])
    count += 1

print('len(diary_list)', len(diary_list)) # diaryと同じになるはず
print('count', count)
print('DIARY', DIARY)

# utility func
def fr(x:np.array):
    count__ = 0
    LL = 0
    # x_mezzo_length, x_micro_length, x_level, x_point, x_hachiko, x_chuo, x_minami = x
    ##### x_mezzo_length, x_micro_length, x_point, x_hachiko, x_chuo, x_minami = x
    x_mezzo_length, x_level, x_point, x_hachiko, x_chuo, x_minami = x
    for i in range(DIARY): # diaryデータから読み込む
        if not diary_list[i]: # 条件満たさないやつはdiary_list内を空にしてるので飛ばせる
            continue
        
        ## 実績値（観測値）
        i_list = diary_list[i]
        dist_car_hachi = i_list[0]/100
        dist_car_chuo = i_list[1]/100
        dist_car_minami = i_list[2]/100
        dist_car_newminami = i_list[3]/100

        dist_dest_hachi = i_list[4]/1000
        dist_dest_chuo = i_list[5]/1000
        dist_dest_minami = i_list[6]/1000
        dist_dest_newminami = i_list[7]/1000

        level_hachi_dum = i_list[8]
        level_chuo_dum = i_list[9]
        level_minami_dum = i_list[10]
        level_newminami_dum = i_list[11]

        point_gate = i_list[12]
        point_amount = i_list[13]/10

        # u_hachiko = np.exp((dist_dest_hachi) * x_mezzo_length + (dist_car_hachi) * x_micro_length + x_hachiko  + level_hachi_dum * x_level) 
        # u_chuo = np.exp((dist_dest_chuo) * x_mezzo_length + (dist_car_chuo) * x_micro_length + x_chuo  + level_chuo_dum * x_level)
        # u_minami = np.exp((dist_dest_minami) * x_mezzo_length + (dist_car_minami) * x_micro_length + x_minami + level_minami_dum * x_level)
        # u_newminami = np.exp((dist_dest_newminami) * x_mezzo_length + (dist_car_newminami) * x_micro_length + level_newminami_dum * x_level)
        
        # u_hachiko = np.exp(np.log(dist_dest_gate) * x_mezzo_length + np.log(dist_car_gate) * x_micro_length + x_hachiko) # + level_dummy * x_level) 
        # u_chuo = np.exp(np.log(dist_dest_gate) * x_mezzo_length + np.log(dist_car_gate) * x_micro_length + x_chuo) # + level_dummy * x_level)
        # u_minami = np.exp(np.log(dist_dest_gate) * x_mezzo_length + np.log(dist_car_gate) * x_micro_length + x_minami) # + level_dummy * x_level)
        # u_newminami = np.exp(np.log(dist_dest_gate) * x_mezzo_length + np.log(dist_car_gate) * x_micro_length) # + level_dummy * x_level)
        
        ### microなし
        u_hachiko = np.exp((dist_dest_hachi) * x_mezzo_length + x_hachiko  + level_hachi_dum * x_level) 
        u_chuo = np.exp((dist_dest_chuo) * x_mezzo_length + x_chuo  + level_chuo_dum * x_level)
        u_minami = np.exp((dist_dest_minami) * x_mezzo_length + x_minami + level_minami_dum * x_level)
        u_newminami = np.exp((dist_dest_newminami) * x_mezzo_length + level_newminami_dum * x_level)


        if point_gate == 1: # ハチ公
            u_hachiko *= np.exp(point_amount * x_point)
        elif point_gate == 2: # 中央
            u_chuo *= np.exp(point_amount * x_point)
        elif point_gate == 3: # 南口
            u_minami *= np.exp(point_amount * x_point)
        ## 新南にはポイントつかない（設定なし）

        deno = u_hachiko + u_chuo + u_minami + u_newminami

        P_hachiko = u_hachiko / deno
        P_chuo = u_chuo / deno
        P_minami = u_minami / deno
        P_newminami = u_newminami / deno

        P_hachiko = (P_hachiko == 0) * 1 + (P_hachiko != 0) * P_hachiko #  .where(P_hachiko != 0, 1)
        P_chuo = (P_chuo == 0) * 1 + (P_chuo != 0) * P_chuo
        P_minami = (P_minami == 0) * 1 + (P_minami != 0) * P_minami
        P_newminami = (P_newminami == 0) * 1 + (P_newminami != 0) * P_newminami

        true_val = df_diary.loc[i, 'd_gate'] if odindex == 'o' else df_diary.loc[i, 'o_gate'] # 正解値
        if true_val == 1:
            LL += np.log(P_hachiko)
        elif true_val == 2:
            LL += np.log(P_chuo)
        elif true_val == 3:
            LL += np.log(P_minami)
        elif true_val == 4:
            LL += np.log(P_newminami)
        else:
            print('ちゃんとした改札から出てない, user', userid)
            continue
        count__ += 1
    
    #print(f'x={x}, sample={count__}, LL={LL}')
    return LL

def mf(x: np.array) -> float:
    return -fr(x)

# hessian
def hessian(x: np.array) -> np.array:
    h = 10 ** -4  # 数値微分用の微小量
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

#t-val
def tval(x: np.array) -> np.array:
    return x / np.sqrt(-np.diag(np.linalg.inv(hessian(x))))

# optimization
x0 = np.zeros(x_dim) #初期値
res = minimize(mf, x0, method="Nelder-Mead")
# res = minimize(mf, x0, method="BFGS")
print(res)

#t値
print(f"tval: {tval(res.x)}")

#初期尤度
L0 = fr(x0)
print(f"L0: {L0:.4f}")

#最終尤度
LL = -res.fun
print(f"LL: {LL:.4f}")

# 決定係数
R = 1 - LL/L0
print(f"R: {R:.4f}")

# 修正済み決定係数
R_adj = 1 - (LL - len(x0))/L0
print(f"R_adj: {R_adj:.4f}")
print('odindex', odindex)

data_n = {# 'parameter': ['mezzo_length', 'micro_length', 'point', 'hachiko', 'chuo', 'minami'], # 
        # 'parameter': ['mezzo_length', 'micro_length', 'level', 'point', 'hachiko', 'chuo', 'minami'],
        'parameter': ['mezzo_length', 'level', 'point', 'hachiko', 'chuo', 'minami'],
        'est.': [f"{num:.2f}" for num in res.x.tolist()],
        't-val': [f"{num:.2f}" for num in tval(res.x).tolist()]}

# n+1 番目以降のデータ
data_n_plus_1 = {'parameter': ['sample size', 'LL0', 'LL', 'rho^2', 'adjusted rho^2'],
                 'est.': [count, f"{L0:.4f}", f"{LL:.4f}",f"{R:.4f}", f"{R_adj:.4f}" ],
                 't-val': [None, None, None, None, None]}

# データフレームを作成
df = pd.DataFrame(data_n)
df = df.append(pd.DataFrame(data_n_plus_1), ignore_index=True)

print(df)