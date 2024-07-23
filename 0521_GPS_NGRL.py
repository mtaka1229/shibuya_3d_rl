# 0521~GPD観測-時間構造化正規化RL同時推定
# 尤度最大化をemで繰り返すOyama & Hato(2018)と同手法
import pandas as pd
import os
import numpy as np
import math
from scipy.stats import rayleigh
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.integrate import quad
import heapq
import time

start_time = time.time()
print('start!')

base_path = '/Users/takahiromatsunaga/res2023' # PPcameraTG/PP/PP_gps/new/10029_20230110/t_loc_data.csv
# df_diary = pd.read_csv(os.path.join(base_path, 'PPcameraTG/PP/PP_webq/q_ans/diary.csv'), encoding='shift-jis')
df_gps_folder = os.path.join(base_path, 'PPcameraTG/PP/PP_gps/shibuya_mezzo_gps3') 
file_names = os.listdir(df_gps_folder) 
df_gps_list = [pd.read_csv(os.path.join(base_path, df_gps_folder, file_name), encoding='shift-jis') for file_name in file_names if file_name.endswith('csv')] 
# print(df_gps_list[0])
df_link = pd.read_csv(os.path.join(base_path, 'shibuya_nw/map/shibuya_mezzo_link.csv'))
df_node = pd.read_csv(os.path.join(base_path, 'shibuya_nw/map/shibuya_mezzo_node.csv'))
df_network = pd.read_csv(os.path.join(base_path, 'shibuya_nw/map/shibuya_mezzo_nodebased_network.csv')) # nodeの接続状況
L = len(df_link)
N = len(df_node) #吸収ノードのところは無くしておく？？
linkid_list = sorted(df_link['id'].unique()) 
nodeid_list = sorted(df_node['id'].unique()) 
OD = len(df_gps_list) # ユニークトリップ数

abs_id = 753 # 吸収状態を表す仮想ノードのid 
T = 30 # transit limitation
w = 1.2 # 1m/sec
length_init = -1
width_init = 1
beta_init = 1
sigma = 15 # 観測パラメータ，最初は固定で
epsilon = 1e-10  # 非常に小さい値を設定（0を取って欲しくない変数xに対してx = np.maximum(pl, epsilon) 

# NWデータから接続行列作る ######### ここ要確認
I = np.zeros((N, N)) # nodeの接続行列．node aとnode kが接続するときI[k ,a] = 1
for i in range(len(df_network)):
    kn = df_network.loc[i, 'k'] # node id 
    an = df_network.loc[i, 'a'] # node id 
    k = nodeid_list.index(kn)
    a = nodeid_list.index(an)
    I[k, a] = 1

### いったん滞在とか考えずに実施

# 変数行列を作る（node_baseなのでnodek_a間のリンクの説明変数をMkaに持つような行列．np.expにかけてから効用関数に入れる！
# 吸収ノード周りは全て0のままになるのでこれでOK
length_mat = np.zeros((N, N)) # nodeからnodeへの遷移の際の説明変数行列
width_mat = np.zeros((N, N))
cross_mat = np.zeros((N, N))
elev_mat = np.zeros((N, N))

for i in range(len(df_network)):
    kn = int(df_network.loc[i, 'k']) # これはid
    an = int(df_network.loc[i, 'a'])
    k = nodeid_list.index(kn) # node knのindex
    a = nodeid_list.index(an)
    kan = None ## 以降でif文がpassされる時，kanは未定義のまま参照されることになる．これだとダメなので最初に定義しておく．
    if not df_link[(df_link['onode'] == kn) & (df_link['dnode'] == an)].empty:
        kan = df_link[(df_link['onode'] == kn) & (df_link['dnode'] == an)]['id'].iloc[0] # リンクデータは無向リンクで考えているので双方向を考える
    if not df_link[(df_link['onode'] == an) & (df_link['dnode'] == kn)].empty:
        kan = df_link[(df_link['onode'] == an) & (df_link['dnode'] == kn)]['id'].iloc[0] # 
    if not kan == None:
        ka = linkid_list.index(kan) # linkidがkanのリンクのindex
        length_mat[k, a] = (df_link.loc[ka, 'length']) # kaはindexなのでこのままでOK
        width_mat[k, a] = (df_link.loc[ka, 'width'])
        cross_mat[k, a] = (df_link.loc[ka, 'cross_dummy'])
        elev_mat[k, a] = (df_link.loc[ka, 'elev_delta']) # このノリで各変数について行列を作っていけばいい

# 緯度経度から距離を計算する関数（経度，緯度の順番）
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

# xがデータフレームかどうかを判定する関数
def is_dataframe(x):
    return isinstance(x, pd.DataFrame)

## Dijkstra法 # 始点oriから各ノードまでの最短距離を出力
def dijkstra(adj, orin): # adjはノード間distを格納してる行列!!．lenght_matを使えばいい． # node id をinputにする
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

def avail_labeler(o_node, d_node, time_limit, w):
    dist_limit = time_limit * w # sec*m/sec=m # time_limitはsecondsで与える
    # print('dist_limit', dist_limit) 
    _, min_cost_o = dijkstra(length_mat, o_node) # oからそのノードまでの最短距離 # min_cost_oはfloat
    _, min_cost_d = dijkstra(length_mat, d_node) # dからそのノードまでの最短距離
    for i in range(N):
        #print(i, N)
        #print(min_cost_o, min_cost_d) #
        min_cost_od = [(min_cost_o[i] + min_cost_d[i]) for i in range(N)] # 別にintにする必要ないか
        # 各ノードが利用可能かの判別
        avail_nodes = [0 if value >= dist_limit else 1 for value in min_cost_od] # 距離制限を超えていたら0とする．そうでなければ利用可能なので1
    return avail_nodes

def find_closest_node(lat, lon, df_node):
    distances = []
    for index, row in df_node.iterrows():
        node_lat = row['lat'] ### df_nodeのlat, lonが逆になってたのでやり直し
        node_lon = row['lon']
        distance = Dis(node_lon - lon, node_lat - lat)
        distances.append(distance)
    closest_node_index = np.argmin(distances)
    return df_node.loc[closest_node_index, 'id']

# 個人の情報を整理 OD: 個人数
dfi_dict = {}
for od in range(OD):
    dfi = df_gps_list[od]
    ti0 = pd.to_datetime(dfi.loc[0, '記録日時']) # 個人の最初の観測時刻
    til = pd.to_datetime(dfi.loc[len(dfi)-1, '記録日時']) # 個人の最後の観測時刻
    ti_limit = (til - ti0).total_seconds() # secに変換
    start_lat = dfi.loc[0, '緯度']
    start_lon = dfi.loc[0, '経度']
    end_lat = dfi.loc[len(dfi)-1, '緯度']
    end_lon = dfi.loc[len(dfi)-1, '経度']
    o_node = find_closest_node(start_lat, start_lon, df_node) # 本当はweb diaryから取るべきだが,,,
    d_node = find_closest_node(end_lat, end_lon, df_node)
    avail_label = avail_labeler(o_node, d_node, ti_limit, w = 1)
    al_rowvector = np.array(avail_label).reshape(1, -1) # 明示的に行ベクトル変換
    al_mat = np.tile(al_rowvector, (N, 1))
    Id = I * al_mat # 利用可能性を考慮した静的な接続行列
    dfi_dict[od] = (o_node, d_node, ti_limit, ti0, til, avail_label, Id) # タプル形式で

#### 確認 #### 
# test1 = avail_labeler(743, 143, 6, 80) # 単位：分, 分速80m # これは個人に対して
# indices = np.where(np.array(test1) == 1)[0] # index
# test_res = [nodeid_list[i] for i in indices] # これでidを返すようになる
# print(test_res) ### 多分OK

### 観測モデル（dfと遷移確率行列pをinputとしてベイジアンによって尤度最大リンク列をreturnする関数）
def Mmodel(x, sigma):  # 個々人に対して実施した方が並列化するのでいいはず # xはすでにpの計算で使ってる # pはnode k→aへの遷移確率
    p = newPall(x) ## 今ここで
    transit_result = []
    for od in range(OD):
        df_transit_result = pd.DataFrame(columns=['t', 'userid', 'k', 'a'])
        # t=1から順に候補リンク集合
        dfi = df_gps_list[od] # 個人の観測列
        user_id = int(dfi.loc[0, 'ユーザーID'])
        # 個人ごとにデータ取得
        onode, dnode, time_limit, ti0, til, avail_label, Id = dfi_dict[od] # node idなど
        onode_idx = nodeid_list.index(onode)  # indexを取得
        dnode_idx = nodeid_list.index(dnode)
        _ta = ta_ = ti0 # 初期値
        rest_time = time_limit # 残り時間の初期値は制限時間一杯
        nonzero_indices = list(np.where(np.array(avail_label) == 1)[0]) # index これタプル
        avail_res = [nodeid_list[i] for i in nonzero_indices] # idを返す．これが利用可能な全ノード集合になる
        print(f'userid: {user_id}, onode: {onode}, dnode: {dnode}, timelimit: {time_limit}, avali_label数: {len(nonzero_indices)}')

        ## もし利用可能ノードないなら飛ばすしかない
        if not nonzero_indices:
            print(f'利用可能ノード無しなので飛ばす')
            transit_result.append(df_transit_result)
            continue

        current_id = onode # 現在地node id （初期値はonode）
        current_idx = onode_idx # 現在地node idx
        for t in range(T): # 割と大きなTを取っておく？こうすることでしか後ろ向き帰納法と両立できなそう．時間でなくて遷移回数の上限．tには意味がない
            if _ta < til:
                current_lon = df_node[df_node['id'] == current_id]['lon'].values[0] 
                current_lat = df_node[df_node['id'] == current_id]['lat'].values[0] 
                current_loc = np.array([current_lat, current_lon])    
                
                cand_nodes = np.where(Id[current_idx, :] == 1)[0] # 接続条件の確認（個人ごとのIdを用いる）．index
                rest_time = (til - _ta).total_seconds() # このタイミングでの残り時間
                avail_label_array = np.array(avail_labeler(current_id, dnode, rest_time + 100, w)) 
                # [o, 1]の2値行列 # time以内にoから向かえ，かつdに向かえるノード集合 ## 緩和する？？？
                avail_label_t = np.where(avail_label_array == 1)[0] # index
                cand_nodes_idx = [item for item in cand_nodes if item in avail_label_t]
                if not cand_nodes_idx:
                    if current_id == dnode:
                        print('dnodeに到着!!!!')
                        new_row2 = {'userid': user_id, 't': t+1, 'k': dnode, 'a': abs_id} 
                        new_row2_df = pd.DataFrame([new_row2])
                        df_transit_result = pd.concat([df_transit_result, new_row2_df], ignore_index=True)
                        current_id = abs_id # スタンプ更新
                        _ta = til # 終了させる
                        #transit_result.append(df_transit_result)
                        continue
                    # print(f'current{current_id}, dnode{dnode}')
                    elif Id[current_idx, dnode_idx] == 1:
                        print('current_idをdnodeにしたい')
                        current_id = dnode
                        _ta = til # 終了させる
                    else: 
                        somosomo = np.where(Id[current_idx, :] == 1)[0] # これは0になり得ないと思うのだが
                        print('ここここここまった', somosomo)
                    continue

                pm_mat = np.zeros((N, N)) # 遷移先のノードの観測尤度の入れ物 # ここに補正ずみの尤度を入れていく # 後から参照しないので1回ループごとに更新
                pam_mat = np.zeros((N, N)) # pm_matとpからdenoとって正規化したやつ

                for cand_node_idx in cand_nodes_idx: # current_idxと接続してる前ノードについて計算
                    cand_id = nodeid_list[cand_node_idx] 

                    if not cand_id in avail_res:
                        print(f'*********これはあり得ないはず**********')

                    elif cand_id in avail_res: # 利用可能なら ## これは100%実行のはず
                        cand_idx = nodeid_list.index(cand_id)  
                        cand_lon = df_node[df_node['id'] == cand_id]['lon'].values[0] 
                        cand_lat = df_node[df_node['id'] == cand_id]['lat'].values[0] 
                        cand_loc = np.array([cand_lat, cand_lon])  ## arrayつけないと引き算できない
                        # 長さを取得してデータを切る
                        #print('cand_loc', cand_loc)
                        link_lengh = length_mat[current_idx, cand_idx] 
                        link_time = link_lengh / w  # w not eq 0, # float 
                        link_time_timedelta = pd.to_timedelta(link_time, unit='s')  # link_time が秒単位の場合
                        ta_ = _ta + link_time_timedelta   
                        sigma_eta = link_lengh / 4 # 適当 # BLEの時は20mで5m，大体length/4でいいのでは
                        dfi['記録日時'] = pd.to_datetime(dfi['記録日時'])
                        
                        if dfi[(dfi['記録日時'] >= _ta) & (dfi['記録日時'] <= ta_)].empty:
                            print(f'userid: {user_id}, t={t}, node{cand_id}の場合，該当時刻に観測データなし,,,') # むちゃ短いリンクだとないかも？
                            continue # continueでいいのか？
                        
                        dfil = dfi[(dfi['記録日時'] >= _ta) & (dfi['記録日時'] <= ta_)] 
                        dfil = dfil.reset_index(drop=True)
                        pl = 1 # このノードの観測尤度
                        result_list = []
                        
                        if len(dfil) > 0: # これは確実に実行されるはず
                            for j in range(len(dfil)): # 全データについてみていくが，
                                tau = pd.to_datetime(dfil.loc[j, '記録日時']) # データjの記録時刻
                                lat_hat = dfil.loc[j, '緯度'] # 観測点
                                lon_hat = dfil.loc[j, '経度'] 
                                x_hat = (lat_hat, lon_hat)  # 観測点
                                delta_j = (tau - _ta).total_seconds() * w 
                                x_j = current_loc + delta_j * (cand_loc - current_loc) # 推定滞在位置(lon, lat)
                                #print('x_hat, x_j, current_loc, cand_loc, del*del', x_hat, x_j, current_loc, cand_loc, delta_j * (cand_loc - current_loc))
                                def integrand(gamma): # 線積分，gammaが0→1で線上を移動で
                                    x_j_bar = gamma * current_loc + (1-gamma) * cand_loc # 等速直線運動の予想点
                                    lat_error, lon_error = abs(x_hat - x_j_bar)
                                    #print('laterror, lonerror', lat_error, lon_error) #lon_errorがめちゃでかいなんで？？？？
                                    dis = Dis(lon_error, lat_error) 
                                    #print('dis', dis)
                                    f_N = rayleigh.pdf(dis, scale = sigma) # 水平誤差がレイリー分布に従うs
                                    d2 = np.linalg.norm(x_j_bar - x_j) 
                                    f_NN = norm.pdf(d2, 0, sigma_eta) ## 真位置と等速直線運動からの予想点の差d2が，平均0分散sigma_etaの正規分布に従う
                                    #print('fN, fNN', f_N, f_NN)
                                    return f_N * f_NN 
                                
                                #######ここにリンク長を掛ける必要がある！
                                result, error = quad(integrand, 0, 1)  # 線積分 
                                
                                #print('result', result)
                                result_list.append(result)
                                # pl *= result # jごとに積み上げる 
                            result_arrayy = np.array(result_list)
                            #print('候補列' , cand_id , result_arrayy)
                            # pl = np.maximum(pl, epsilon)  # plの全ての要素がepsilon以上になるようにする
                            pl_corrected = np.mean(result_arrayy)
                            # if user_id == 23171:
                            # print(f'candnode{cand_id}の平均尤度{pl_corrected}')
                            # pl_corrected = pl ** (-(len(dfil))) ##### 観測点1点あたりの平均的な確率を計算→リンク長に応じてpの乗じる回数が異なるので不平等

                        else:
                            print('dfilの長さが0, 変')
                            pl_corrected = 1

                        #print(f'candnode{cand_id}の平均尤度{pl_corrected}')
                        pm_mat[current_idx, cand_idx] = pl_corrected

                    else: 
                        print('ここくることある？？')
                        continue # ダメなら次のループへ
                
                pam_mat[:, :] = pm_mat[:, :] * p[od, t, :, :] # pはあらかじめ入れる（引数）．時変なのでtで指定される
                deno = np.sum(pam_mat[:, :], axis=1) # np.sum(pmstmatrix[t-1], axis=1)になってた
                deno = (deno == 0) * 1 + (deno != 0) * deno ## 観測尤度の分母
                deno_rep = np.tile(deno, N).reshape(N, N).T ##### これがないと変になった
                pam_mat[:, :] = pam_mat[:, :] / deno_rep ## これが観測尤度

                nonzero_indices2 = np.nonzero(pam_mat)
                nonzero_values2 = pam_mat[nonzero_indices2]
                # インデックスと値を表示
                # for index, value in zip(zip(nonzero_indices2[0], nonzero_indices2[1]), nonzero_values2):
                #     index2 = zip(nodeid_list[index[0]], nodeid_list[index[1]])
                #     print(f"Index: {index2}, Value: {value}") # node k→aの観測尤度が出る
                # print('pam_mat', pam_mat)
                for i in range(len(nonzero_values2)):
                    index = (nonzero_indices2[0][i], nonzero_indices2[1][i])
                    index2 = (nodeid_list[index[0]], nodeid_list[index[1]])
                    value2 = nonzero_values2[i]
                    print(f"Index: {index2}, Value: {value2}") # node k→aの観測尤度が出る
                
                #print(f'nonzeromat length{len(nonzero_values2)}') ####  なんでかこれ全部0になってる
                max_node_idx = np.argmax(pam_mat[current_idx, :]) # 観測尤度最大nodeのindexになる．
                maxl_link_length = length_mat[current_idx, max_node_idx]
                maxl_passtime = pd.to_timedelta(maxl_link_length / w, unit='s') # float→datetimeにする必要がる
                max_node_id = nodeid_list[max_node_idx] # これでnodeidに変換
                # transit_result.append(max_node_id) # 通過ノードリスト（逐次更新）
                new_row = {'userid': user_id, 't': t+1, 'k': current_id, 'a': max_node_id}
                new_row_df = pd.DataFrame([new_row])
                # print(f'newrow{new_row}')
                df_transit_result = pd.concat([df_transit_result, new_row_df], ignore_index=True)

                current_id = max_node_id # スタンプ更新
                current_idx = max_node_idx
                # print(f'_ta = {_ta}, passtime={maxl_passtime}, til{til}')
                _ta = _ta + maxl_passtime # maxnodeで決まるリンク長に応じて時刻変える
                # これで次のループに行く（_taがまだリミットtilを超過してなければ）
            
            # while 文を抜けたら吸収させる
            # 直前のcurrent_idのスタンプで判断
            elif current_id == dnode: # すでにdfでdnodeに辿り着いてるので
                print('dnodeに到着!!!!')
                new_row2 = {'userid': user_id, 't': t+1, 'k': dnode, 'a': abs_id} 
                new_row2_df = pd.DataFrame([new_row2])
                df_transit_result = pd.concat([df_transit_result, new_row2_df], ignore_index=True)
                current_id = abs_id # スタンプ更新

            elif current_id != dnode and current_id != abs_id:
                # dnodeをつけてから吸収させる
                print(f'tは終了してるがdnodeではない, current={current_id}')
                #dnode_idx = nodeid_list.index(dnode)
                # current_it と dnodeが接続しない場合はエラー
                if Id[current_idx, dnode_idx] == 0:
                    print('erorrrrrrrrrrr')
                else:
                    new_row = {'userid': user_id, 't': t+1, 'k': current_id, 'a': dnode}
                    new_row2 = {'userid': user_id, 't': t+1, 'k': dnode, 'a': abs_id} # 仮想ノードを用意しておく！！！！ # abs
                    new_row_df = pd.DataFrame([new_row])
                    new_row2_df = pd.DataFrame([new_row2])
                    df_transit_result = pd.concat([df_transit_result, new_row_df, new_row2_df], ignore_index=True)
                    current_id = abs_id # スタンプ更新
            
            elif current_id == abs_id:
                # あとはひたすらabsを繰り返す
                print('ひたすら吸収')
                new_row = {'userid': user_id, 't': t+1, 'k': abs_id, 'a': abs_id}
                new_row_df = pd.DataFrame([new_row])
                df_transit_result = pd.concat([df_transit_result, new_row_df], ignore_index=True)
        # print(df_transit_result) # 個人の経路情報
        transit_result.append(df_transit_result) 
        # xがデータフレームかどうかを判定する関数
        # print(is_dataframe(df_transit_result))
    return transit_result 

def newPall(x): 
    Pall = np.zeros((OD, T, N, N)) # Mは一定だけど下流効用が時変なのでpも時変か
    prebeta = np.full((N, N), x[-1])
    # beta = np.ones((N, N)) # リンクごとに与えられる．つまりnode k→aに対して1つづつ対応
    beta = prebeta ** (length_mat/w) # これでいいらしい．多分 # 繋がってなければlength_matは0になる #### 0^0=1, 0^a=0 
    ### length_matは接続してなければ0になる．
    # print(f'{prebeta[0, 0], (length_mat/w)[0, 0], prebeta[0, 0] ** (length_mat/w)[0, 0]}')
    # print(f'{prebeta[0, 1], (length_mat/w)[0, 1], prebeta[0, 1] ** (length_mat/w)[0, 1]}')
    # print(f'len/w{length_mat/w}')
    # print(f'print beta[n,n]{beta[0, :]}') #### length_matは機能してる
    for od in range(OD):
        _, _, _, _, _, _, Id = dfi_dict[od] # node idなど
        M = np.zeros((N, N)) # Mは時不変にせざるを得ない．
        # for ts in range(T-1): # 説明変数直接かけることでMset不要に
            # Mts = Id[ts, :, :] * np.exp(length_mat ** x[0]) * np.exp(width_mat ** x[1])  # Mset(x)[ts, :, :] # N*Nの形は同じなので
            # M[ts, :, :] = Mts
        M =  Id * np.exp(length_mat/100 * x[0] + width_mat/10 * x[1]) 
        #print('v[0,1]', (np.exp(length_mat/100 * x[0] + width_mat/10 * x[1])[0, 1]))
        #print('Id[0,1]', Id[0, 1]) # なんか0になってる
        #print('M[0,:]', M[0,:], Id[0, :]) # IdはOK # Mはダメだ，接続してるのが0でそれ以外はnanになる
        #print(f'explength{np.exp(length_mat /100 * x[0])}')
        #print(f'widthmat{np.exp(width_mat/10 * x[1])}')
        z = np.ones((T, N)) 
        for t in range(T-1, 0, -1):
            zii = M * (z[t, :] ** beta)  # ここで各リンクの距離に応じた行列をかけてやることでNGRLになるはず！！s．多分要素ごとに掛けれてる # zはbroadcastされる
            zi = zii.sum(axis = 1)
            z[t-1, :] = (zi==0)*1 + (zi!=0)*zi
        for t in range(T-1): 
            for k in range(N):
                for a in range(N):
                    if M[k, a] == 0: # 接続してなければ0
                        continue 
                    Pall[od, t, k, a] += np.exp(np.log(M[k, a]) + beta[k, a] * np.log(z[t+1, a]) - np.log(z[t, k])) 
                    ### ここもbetaにリンク長比例のベクトルをかけることで対応
                    # 謎にエラー吐いたのでbeta[k, a]で修正
    return Pall

def loglikelihood(x): # 個人で並列化した方がいいが，，，
    global transit_result # 宣言することで関数外で定義された変数を参照可能
    LL = 0        
    Pt = newPall(x)
    for od in range(OD):
        dfi_result = transit_result[od] # transit_resultはグローバル変数的に与える
        a_list = dfi_result['a'].tolist()
        if dfi_result.empty:
            print("dfi_result is empty")
            continue
        userid = dfi_result.loc[0, 'userid']
        print(f'userid{userid}')
        if abs_id not in a_list:
            print('データフレーム不完全なのでカット')
            continue
        ### dfi_resultはdataframeな気がする
        # print(f'dfi_result{dfi_result}')

        # 遷移回数はTにする，せざるを得ない，時間ではなく遷移回数．ただbetaは距離ベースで計算できているのでOKのはず，，，
        for t in range(T):
            kn = int(dfi_result.loc[t, 'k']) # id
            an = int(dfi_result.loc[t, 'a']) # id
            k = nodeid_list.index(kn)
            a = nodeid_list.index(an)
            pka = Pt[od, t, k, a]
            pka = (pka == 0) * 1 + (pka != 0) * pka 
            print(pka)
            LL += np.log(pka)
    print(f'x={x}でLL={LL}')
    return -LL 

## 推定 ##
x_init = np.zeros(3)
xbounds = [(-10, 10), (-10, 10), (0, 1)] #, (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 1)]
n = 0
dL = 100

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
    print(f'hesse行列の逆行列{np.linalg.inv(hessian(x))}')         
    print(f'各パラメタの分散{-np.diag(np.linalg.inv(hessian(x)))}')      
    return x / np.sqrt(-np.diag(np.linalg.inv(hessian(x))))

x0 = x_init
x0[0] = length_init
x0[1] = width_init
x0[-1] = beta_init
b0 = sigma # sigma = 20

# while dL >= 10:
print(f'<<<at {n}th loop>>>')

# e-step # 今は固定
# global変数    
transit_result = Mmodel(x0, sigma) # Mmodel一回回すのに5分程度

#df_res_all = link_sequence(x0, b0)
x = x0

# m-step
res = minimize(loglikelihood, x, method='L-BFGS-B', bounds = xbounds) #, options={"maxiter":10,"return_all":True}) 
x0 = res.x

dL = np.sum(np.abs(x - x0))
print('x0=', x0)
print('lnLL=', -1 * res.fun)
print('dL=', dL)
n += 1

print('while終わり')

tval = tval(x0)
L0 = -1 * loglikelihood(x_init)
LL = -1 * res.fun
end_time = time.time()
proc_time = end_time - start_time

###### 最終結果の出力 ######
print("計算時間")
print(proc_time)
print("Inputdata")
#print(search_folder)
print("結果の表示")
print(res)
#print("説明変数の表示")
#print(linkv)
print("初期尤度 = ", L0)
print("最終尤度 = ", LL)
print("ρ値 = ", (L0 - LL) / L0)
print("修正済ρ値 = ", (L0 - (LL - len(x0))) / L0)
print("パラメータ初期値 = ", x_init)
print("パラメータ推定値 = ", x0)
print("時間割引率 = ", x0[-1])
print("sigma = ", b0)
print("t値 = ", tval)