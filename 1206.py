
###### 各linkのoとdの座標を入れておく配列 ###### # 線分とビーコンの距離に関しての情報は向きがあってもなくても変わらない
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

###### 各ビーコンの座標を入れておく配列 ###### 
beacon_loc_array = []
for j in range(len(df_ble)):
    beacon_id = df_ble.loc[j, 'ID']
    x_j = df_ble[df_ble['ID'] == beacon_id]['x'].iloc[0]
    y_j = df_ble[df_ble['ID'] == beacon_id]['y'].iloc[0]
    z_j = df_ble[df_ble['ID'] == beacon_id]['floor'].iloc[0]
    beacon_loc = np.array([x_j, y_j, z_j])

    beacon_loc_array.append(beacon_loc) # beacon_loc_array[i]がビーコンi+1の位置座標



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
dens_jt = np.zeros((23, 45), dtype = int)
max_jt = np.zeros((23, 45), dtype = int)
mean_jt = np.zeros((23, 45), dtype = int)
#count_jt = np.zeros((23, 45), dtype = int)

grouped = df_quater.groupby(['ID', 'timestep']) # listにして渡す必要がある
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

####### リンク接続行列（無向） ####### 
def I_nod(link_data): 
    n = len(link_data)
    I = np.eye(n) # 滞在リンク（吸収リンクにもつけて良い）
    for i in range(n):
        O = link_data.loc[i, 'o'] # 当該linkのOnode
        D = link_data.loc[i, 'd'] # 当該linkのDnode
        for j in range(n): # リンクiのOorDをOないしDとしているリンクjとは接続
            if ((link_data.loc[j, 'o'] == O) or (link_data.loc[j, 'o'] == D)) or (link_data.loc[j, 'd'] == O) or (link_data.loc[j, 'd'] == D): 
                I[i, j] = 1
    return(I)

I = I_nod(df_link)

###### 階段リンクについて別途処理 ###### # i = 26~44
for i in range(26, 45, 2):
    # 階段には滞在リンク入れない．向きは考えないので面倒なことは考えない
    I[i-1, i-1] = 0
    I[i, i] = 0

###### 吸収リンクからは吸収リンクにしか接続しない
for i in range(46, 49):
    I[i-1, :] = 0 # 他には繋がらない
    I[i-1, i-1] = 1 # 自分自身にはつながる # 時間構造化するなら自分自身への滞在は可能にする!!!!!とりあえず構造化でできるかチャレンジする

###### 改札の吸収リンク ######
ddata = [46, 47, 48]
D = len(ddata)

####### パラメタ初期設定 #######
xdim = 2
x = np.zeros(xdim)
x[0] = -1
x[-1] = 0.8
sigma = 5 #### 想像できなすぎる．理想は，リンクごとに設定して推定

###### タイムステップ設定 ######
T = 11 # 10stepで最終リンクには辿り着く人たち，吸収に行くのに+1timestep要すると考えるとT=11が妥当か．
###################### inputdataもtimestep10程度以下のものを持ってくる必要があるね #########


####### 観測モデル ######

####### 観測方程式 ######
def qq(dist):
    return (dist/((sigma)**2) + np.exp(-((dist)**2)/2/((sigma)**2)))

# 重み付き最小二乗法を最小化する関数
def weighted_least_squares(coordinates, observations):
    # 各観測点の座標と重みを取り出す
    observed_coordinates = observations[:, :-1]  # 最後の列（重み）を除いた座標
    weights = observations[:, -1]  # 重み
    
    # 予測座標と観測座標の誤差の計算
    errors = coordinates - observed_coordinates
    
    # 重みを考慮した重み付き誤差の二乗和を返す
    weighted_errors = weights * np.sum(errors ** 2, axis=1)
    
    return np.sum(weighted_errors)

def measuring_model(x):
    print(f'パラメタ{x}で観測モデル回す')
    measured_loc = np.zeros(6) # 'userid', 'timestep', 'x', 'y', 'z' 
    for file_name in file_list:
        file_path = os.path.join(search_folder, file_name)
        dfi = pd.read_csv(file_path)
        
        base_name = file_name.split('.')[0]   
        userid = int(base_name)

        nts = dfi['timestep'].nunique()                
        grouped = dfi.groupby('timestep')
        dfi_list = [group.reset_index(drop=True) for name, group in grouped]  

        i_res = np.zeros(6)
            
        for t in range(nts): # tが最後の時は次の最大ビーコンの座標を参照できない
            dfit = dfi_list[t]
            ts = int(dfit.loc[0, 'timestep'])

            result_beacon = dfit.groupby('ID')
            result_beacon_list = [group.reset_index(drop=True) for name, group in result_beacon]

            tdict = {}

            for j in range(len(result_beacon_list)): # 各ビーコンの結果を束ねたdataframe
                dfitj = result_beacon_list[j]
                jscore = 0 # 初期値は0にしておく
                jid = dfitj.loc[0, 'ID'] # このビーコンのid
                max_rssi = max_jt[jid-1, ts-1] # tsでのbeaconのmaxのrssi

                for k in range(len(dfitj)): 
                    rssi = dfitj.loc[k, 'RSSI']                        
                    jscore += max_rssi / rssi 
                    tdict[jid] = jscore  
            
            # dictにビーコンのidとRSSI実績値による重みが入った
            # 最大ビーコンのidをメモ
            jmax = max(tdict, key = tdict.get)

            if t < nts-1:
                if len(tdict) > 1: # 2点以上で観測されてたら，最小二乗法で一点の推定位置を出すことは可能  
                    observation_array = np.zeros(4)
                    for b in range(len(tdict)):
                        beacon_i = list(tdict.keys())[b] # beacon_iはビーコンのidなので-1してindexにする
                        beacon_i = int(beacon_i)
                        loc_i = beacon_loc_array[beacon_i-1] # 位置座標取得
                        beacon_w = list(tdict.values())[b]

                        result_array = np.insert(loc_i, len(loc_i), beacon_w)
                        observation_array = np.vstack((observation_array, result_array))
                    # 先頭の0の行を削除
                    observation_array = np.delete(observation_array, 0, axis = 0)
                    # n点の位置が取得されてobservation_arrayに入った．weighted_least_squares(coordinates, observations)を使って位置を推定

                    # 初期座標の設定（仮の値を使用）
                    initial_coordinates = np.array([-11920, -37880, 20])

                    # 重み付き最小二乗法を最小化して推定座標を求める
                    result = minimize(weighted_least_squares, initial_coordinates, args=(observation_array,), method='Nelder-Mead')

                    # 推定された座標を取得
                    estimated_coordinates = result.x
                    estimated_coordinates_with_j = np.insert(estimated_coordinates, 0, jmax)
                    estimated_coordinates_with_ts = np.insert(estimated_coordinates_with_j, 0, ts) # tsを列の先頭に追加

                elif len(tdict) == 1: # 1点だったらどうしましょう→次の観測を参照
                    observation_array = np.zeros(4)

                    beacon_i = list(tdict.keys())[0] # beacon_iはビーコンのidなので-1してindexにする
                    beacon_i = int(beacon_i)
                    loc_i = beacon_loc_array[beacon_i-1] # 位置座標取得
                    beacon_w = list(tdict.values())[0]

                    #result_array = np.array(loc_i + beacon_w, dtype=float)
                    result_array = np.append(loc_i, float(beacon_w))
                    observation_array = np.vstack((observation_array, result_array))

                    # 次のmaxbeaconを参照
                    dfitt = dfi_list[t+1]
                
                    next_result_beacon = dfitt.groupby('ID')
                    next_result_beacon_list = [group.reset_index(drop=True) for name, group in next_result_beacon]

                    # 時刻t+1の各ビーコンのscoreを記録するdictの準備
                    ttdict = {}
                    for j in range(len(next_result_beacon_list)): # 各ビーコンの結果を束ねたdataframe
                        dfittj = next_result_beacon_list[j]
                        next_jscore = 0 # 初期値は0にしておく
                        next_jid = dfittj.loc[0, 'ID'] # このビーコンのid
                        next_max_rssi = max_jt[jid-1, (ts+1)-1] # tsでのbeaconのmaxのrssi

                        for k in range(len(dfittj)): # freq = count / count_tot
                            next_rssi = dfittj.loc[k, 'RSSI']                        
                            next_jscore += next_max_rssi / next_rssi # beacon_eval(rssi, max_rssi) # , freq)
                            ttdict[jid] = next_jscore

                    # 次の最大beaconのkeyとscoreを取得して→maxじゃね
                    next_max_key = max(ttdict, key = ttdict.get)
                    next_max_score = ttdict[next_max_key]

                    next_max_beacon_loc = beacon_loc_array[next_max_key-1] # 次の最大ビーコンの座標
                    #next_result_array = np.append(next_max_beacon_loc, float(next_max_score))
                    #next_result_array = np.array(next_max_beacon_loc + next_max_score, dtype=float)

                    next_result_array = np.insert(next_max_beacon_loc, len(next_max_beacon_loc), next_max_score)
                    observation_array = np.vstack((observation_array, next_result_array))

                    # 先頭の0の行を削除
                    observation_array = np.delete(observation_array, 0, axis = 0)
                    # 2点の位置が取得されてobservation_arrayに入った．weighted_least_squares(coordinates, observations)を使って位置を推定

                    # 初期座標の設定（仮の値を使用）
                    initial_coordinates = np.array([-11920, -37880, 20])

                    # 重み付き最小二乗法を最小化して推定座標を求める
                    result = minimize(weighted_least_squares, initial_coordinates, args=(observation_array,), method='Nelder-Mead')

                    # 推定された座標を取得
                    estimated_coordinates = result.x
                    estimated_coordinates_with_j = np.insert(estimated_coordinates, 0, jmax)
                    estimated_coordinates_with_ts = np.insert(estimated_coordinates_with_j, 0, ts) # tsを列の先頭に追加(ts, x, y, z)
            
            elif t == nts-1:
                last_beacon = max(tdict, key = tdict.get)
                last_beacon = int(last_beacon)
                estimated_coordinates = beacon_loc_array[last_beacon-1] # 最後は一番検出が顕著だったビーコンの座標で
                
                estimated_coordinates_with_j = np.insert(estimated_coordinates, 0, jmax)
                estimated_coordinates_with_ts = np.insert(estimated_coordinates_with_j, 0, ts)

            estimated_coordinates_with_ts_i = np.insert(estimated_coordinates_with_ts, 0, userid) # useridを列の先頭に追加
            #print(f'ires{i_res}, estimated_coordinates_with_ts_i{estimated_coordinates_with_ts_i}') # maxbeaconがestimatedの方に入ってない
            i_res = np.vstack((i_res, estimated_coordinates_with_ts_i))

        measured_loc = np.vstack((measured_loc, i_res))
        measured_loc_df = pd.DataFrame(measured_loc, columns = ['userid', 'timestep', 'maxbeacon', 'x', 'y', 'z'])

    return measured_loc_df

##########################
###### 経路選択モデル ######
##########################

###### 時空間プリズム準備 ######
Ilist = []
for i in range(L):
    I[i, i] = 1 # Iは普通の空間接続行列．これに滞在リンクを追加している
    
# 起点Oからの到達可能性指示変数ioの定義
Itlist = np.zeros((T+1, L, L))
II = np.copy(I)
Itlist[0, :, :] = np.eye(L) # ItlistにはT+1成分あるが，その第一成分(t=0)は単位行列ということ．t=1以降はfor内で計算

for ts in range(1, T+1):
    Itlist[ts, :, :] = II # 時刻1では普通の接続行列
    II = np.dot(II, I) # 接続条件の積．つまり2回先に接続していたら1，そうでなければ0
    II = np.where(II > 0, 1, 0) 

# 終点Dからの到達可能性指示変数idの定義
Ittlist = np.zeros((T+1, L, L))
for ts in range(T+1):
    Ittlist[ts, :, :] = np.transpose(Itlist[T - ts, :, :]) 

###### 即時効用行列 ######
def Mset(x): 
    cost = np.exp(df_link['length'] * x[0]) #  + df_link['staire'] * x[1]) # + df_link_integrated['staire_with_esc'] * x[3])
    cost = pd.concat([cost]*L, axis=1)
    cost = cost.T
    cost_numpy = cost.values # DataFrameをNumPy配列に変換
    return cost_numpy

###### 対数尤度関数 ######
def loglikelihood(x):
    beta = x[-1]
    LL = 0
    # newV不要 # z = newV(x) #V = np.log(z)
    
    # 観測モデル回す
    measured_loc_df = measuring_model(x)

    # 観測結果をuseridで分割
    grouped = measured_loc_df.groupby('userid')
    df_list = [group for name, group in grouped]
    df_list = [data.reset_index(drop=True) for data in df_list]    
    
    ###############################
    ######## ODを決める ###########
    saikyo_key = [1, 2, 3, 4, 5, 6]
    yamate_key = [7, 8, 9, 10, 11, 12, 13]
    discriminate_key = [19, 22, 23]
    abs_beacons = [16, 20, 21]

    #### 以後個人ごとに処理
    for i in range(len(df_list)): # 構造化しているので目的地ではなく各自で分ける
        dfi = df_list[i]
        abs = dfi.loc[len(dfi)-1, 'maxbeacon']

        # 吸収がハチ公・中央以外だったら飛ばす
        if abs not in abs_beacons:
            continue
        
        maxbeacon_list = dfi['maxbeacon'].tolist()

        # O決める
        first_beacon = int(dfi.loc[0, 'maxbeacon']) # 最初のビーコン
        first_dist_list = d_array[:, first_beacon-1] 
        first_dist_dict = {index + 1: value for index, value in enumerate(first_dist_list)} # keyはlinkidになっている．max_beaconと各リンクの距離が入ってる

        if any(item in discriminate_key for item in maxbeacon_list): # 埼京オンリーにする
            filtered_dict = {key: first_dist_dict[key] for key in first_dist_dict if key in saikyo_key}
            max_key = min(filtered_dict, key = filtered_dict.get) # これが初期リンクとなる．埼京ホームのうち，最もmax_beaconに近いリンク
        else: # 山手オンリー
            filtered_dict = {key: first_dist_dict[key] for key in first_dist_dict if key in yamate_key}
            max_key = min(filtered_dict, key = filtered_dict.get)
                  
        a0 = int(max_key) 
        tsi = len(dfi)
        # 次にDを確定させにいく
        last_beacon = int(dfi.loc[tsi-1, 'maxbeacon']) # 一番下の行を指定
        last_link = 0
        if last_beacon == 16: # node39
            last_link = 25 # 最後のリンク
            abs_link = 46
        elif last_beacon == 20: # node28
            last_link = 17
            abs_link = 47
        elif last_beacon == 21: # node29
            last_link = 18   
            abs_link = 48   
        else:
            continue  

        ###### OD決まったら時空間プリズム制約Idを入れる #####
        Id = np.zeros((T, L, L))
        ao = a0 - 1             # ODlist.loc[od, 'o'] - 1  ## ODlistではなく直前を参照
        ad = last_link - 1      # ODlist.loc[od, 'd'] - 1

        for ts in range(T):
            if ts == 0:
                Id[ts, ao, :] = I[ao, :] # 最初はoの接続行
                continue

            alist = np.where(Itlist[ts + 1, ao, :] == Ittlist[ts + 1, ad, :])[0]  # +1しない方がいいかも
            for a in alist:
                if Itlist[ts + 1, ao, a] == Ittlist[ts + 1, ad, a]:
                    klist = np.where(I[:, a] == 1)[0]

                    for k in klist:
                        if len(np.where(Id[ts - 1, :, k] == 1)[0]) != 0:
                            Id[ts, k, a] = 1
      
        ##### 即時効用 #####
        M = np.zeros((T, L, L))

        for ts in range(T):
            Mts = Id[ts, :, :] * Mset(x)
            M[ts, :, :] = Mts

        z = np.ones((T+1, L))
        for t in range(T, 0, -1):
            zii = M[t-1, :, :] * (z[t, :] ** beta)
            zi = zii.sum(axis = 1)
            z[t-1, :] = (zi==0)*1 + (zi!=0)*zi
        
        #####################################
        ###### ここから下は書き換える！df_dataに相当するものを作る
        # measured_loc_dfの今見てるやつ（dfi）にk, a, を足していく
        # TSも追加する．吸収にk, aを足していくといい
        # 選択モデルで補正して，最大のものを採用するのが良いか＊
        #####################################
        init_ts = dfi.loc[0, 'timestep']

        ## dfi_dataなるデータフレームを作るのがいいわ．初期化して与える
        dfi_data = pd.DataFrame({'t': range(1, T + 1)})
        # 'k', 'a' 列を追加し、初期化する
        dfi_data['k'] = 0
        dfi_data['a'] = 0

        at_1 = a0 # 初期値的に与える

        for t in range(T): # len(dfi)を超えたら，吸収状態の行を追加していく
            Idt = Id[t, :, :]
            
            if (t < len(dfi)-1):
                
                # at_1 = dfi_data.loc[t-1, 'k'] # t=0の回で更新されるので良い
                a_row = Idt[at_1-1, :] 
                at_cand = np.where(a_row == 1)[0]+1 # at_1のプリズム内のリンク
                if len(at_cand) == 0:
                    break

                est_x = dfi.loc[t, 'x']
                est_y = dfi.loc[t, 'y']
                est_z = dfi.loc[t, 'z']
                est_loc = np.array([est_x, est_y, est_z])

                res_dict = {} 

                for at in at_cand: # atはlinkid
                    p_o = oddslink_loc_array[at-1][0]
                    p_d = oddslink_loc_array[at-1][1]
                    dist = shortest_distance_to_segment(p_o, p_d, est_loc) # np.linalg.norm(est_loc - at_mid_loc)
    
                    pp = np.exp(np.log(M[t, at_1-1, at-1]) + beta * np.log(z[t+1, at-1]) - np.log(z[t, at_1-1]))

                    q = qq(dist)
                    nume = q ** pp 

                    deno = 0
                        
                    at_1_row = I[at_1-1, :] # prime は，at_1からつながっている全リンク
                    at_prime_list = []
                    at_prime_list = np.where(at_1_row == 1)[0]+1 

                    for at_prime in at_prime_list:
                        p_o_prime = oddslink_loc_array[at_prime-1][0]
                        p_d_prime = oddslink_loc_array[at_prime-1][1]
                        dist_prime = shortest_distance_to_segment(p_o_prime, p_d_prime, est_loc) # np.linalg.norm(est_loc - at_mid_loc)
                        q_prime = qq(dist_prime)
                        #dist = d_array[at_prime-1, max_beacon-1]
                        #pmat_prime = mq(dist, min_dist)
                        pp = (np.exp(np.log(M[t, at_1-1, at_prime-1]) + beta * np.log(z[t+1, at_prime-1]) - np.log(z[t, at_1-1])))
                        deno += q_prime * pp
                    if deno == 0:
                        continue

                    delta = nume / deno 
                    #pp = (p[at_1-1, at-1] == 0) + ((p[at_1-1, at-1] != 0) * p[at_1-1, at-1])
                    # atにリンク尤度を入れる．これを辞書にする．valueが最大のものを次のat_1にする
                    # res_dict[at] = delta * np.log(pp) # 観測尤度を入れるのでdeltaだけ入る

                    res_dict[at] = (delta, delta * np.log(pp))

                if not res_dict:
                    print('sss')
                    break

                else:
                    max_delta_key = max(res_dict, key=lambda k: res_dict[k][0]) # deltaが最も大きいリンクのリンクidということ
                    dfi_data.loc[t, 'a'] = max_delta_key
                    at_1 = max_delta_key # これはlinkid．毎回更新される                

                    #LL += res_dict[max_delta_key][1] # 最大リンクのdelta*np.log(pp)


            elif t == len(dfi)-1: # 最後→吸収の行をくわえる
                dfi_data.loc[t, 'k'] = last_link
                dfi_data.loc[t, 'a'] = abs_link


            else: # t > len(dfi)の時は吸収→吸収だね
                dfi_data.loc[t, 'k'] = abs_link
                dfi_data.loc[t, 'a'] = abs_link                

        ####### ここまでで分析用にdfiを加工

        ##### 以上で時刻tにどのリンクを選んだかを確定的に推定し，以下で尤度計算に持ち込む ######
        klink = list(dfi_data.loc[:, 'k'])
        alink = list(dfi_data.loc[:, 'a'])

        for t in range(len(dfi)): # 吸収のところは実質的には関係ないので飛ばして良いのでは
            k = int(klink[t]) - 1 
            a = int(alink[t]) - 1
            
            #pp = np.exp(np.log(M[t, k, a]) + beta * np.log(z[t+1, a]) - np.log(z[t, k]))
            # print(f'個人{od+1}の時刻{t+1}の遷移は{k+1}から{a+1}で，その即時効用は{M[t, k, a]}，{a+1}の期待効用は{z[t+1, a]}')
            
            ##### 吸収の時はest_locとかの概念ない！！！！
            # estimatedは
            est_x = dfi.loc[t, 'x']
            est_y = dfi.loc[t, 'y']
            est_z = dfi.loc[t, 'z']
            est_loc = np.array([est_x, est_y, est_z])

            # q(estimated_x, selected_link) # linkの中点at_mid_locを取得する必要がある
            # 考えるのは遷移先のリンクのリンク尤度なのでa
            p_o = oddslink_loc_array[a][0]
            p_d = oddslink_loc_array[a][1]

            dist = shortest_distance_to_segment(p_o, p_d, est_loc) # np.linalg.norm(est_loc - at_mid_loc)

            # 一応観測方程式と呼べるものはこれか．
            q = qq(dist) # = dist/((sigma)**2) + np.exp(-((dist)**2)/2/((sigma)**2)) # レイリー分布

            k_row = I[k, :]
            a_prime_list = []
            a_prime_list = np.where(k_row == 1)[0]+1


            for a_prime in a_prime_list:
                p_o_prime = oddslink_loc_array[a_prime-1][0]
                p_d_prime = oddslink_loc_array[a_prime-1][1]
                dist_prime = shortest_distance_to_segment(p_o_prime, p_d_prime, est_loc) # np.linalg.norm(est_loc - at_mid_loc)
                q_prime = qq(dist_prime)
                        #dist = d_array[at_prime-1, max_beacon-1]
                        #pmat_prime = mq(dist, min_dist)
                pp = (np.exp(np.log(M[t, k, a_prime-1]) + beta * np.log(z[t+1, a_prime-1]) - np.log(z[t, k])))
                deno += q_prime * pp
            if deno == 0:
                continue

            delta = nume / deno 

            logp = np.log(M[t, k, a]) + beta * np.log(z[t+1, a]) - np.log(z[t, k])

            LL += delta * logp

    return -LL
