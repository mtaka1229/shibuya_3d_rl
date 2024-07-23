# 目的地を絞ったRLでver4を書き直し→これでホームリンク長の負効用が有意に出てほしい
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
# 実態リンクのうち片方向のリンクのみ入れたもの．観測モデルに使う．観測モデルでは仮想リンクや吸収リンクは関係ない上，グラフの向きも判定できないので．各リンクidに+1したものが逆方向リンクに対応
df_link_odds = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_odds.csv")
l = len(df_link_odds)

# ここには仮想リンクと吸収リンクを入れている
df_link_integrated = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_link_pre_integrated.csv')
L = len(df_link_integrated)

# 17:00-17:15からさらに絞ったデータを使用
search_folder = "/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_17/quater_address_gateend500"
file_list = list(os.listdir(search_folder))

# ここは常に変わらない
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/shibuya_node.csv')
df_ble = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_nw_detail/ble_nw.csv')

###### 各linkのoとdの座標を入れておく配列 ######
oddslink_loc_array = [] # 空リストを用意
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
    
    # 垂直ベクトルのノルム（距離）を求める
    distance = np.linalg.norm(v)
    
    # 線分の範囲内であるかチェック
    dot_product = np.dot(v1, segment)
    if dot_product < 0:
        return np.linalg.norm(x - p1)
    elif dot_product > np.dot(segment, segment):
        return np.linalg.norm(x - p2)
    else:
        return distance

###### d_arrayの用意 ######
d_array = np.zeros((len(df_link_odds), len(df_ble))) # 各リンク(奇数側)とbleビーコンとの距離配列

for i in range(len(df_link_odds)):
    p_o = oddslink_loc_array[i][0] # o座標 # i=1の時
    p_d = oddslink_loc_array[i][1] # d座標

    for j in range(len(df_ble)):
        x_ap = df_ble.loc[j, 'x']
        y_ap = df_ble.loc[j, 'y']
        z_ap = df_ble.loc[j, 'floor']*1000 # zの値をめちゃ大きくすることで，階を挟んだ電波捕捉を捨象（階の厳密な高さが不明なのと，床板・天井版による電波遮蔽を考慮するため）．階が同じなら0になるので変な影響はない
        p_ap = np.array([x_ap, y_ap, z_ap])

        d_array[i, j] = shortest_distance_to_segment(p_o, p_d, p_ap)

###### d_node_arrayの用意 ######
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

####### リンク接続行列（無向） ####### 
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

####### リンク接続行列（有向） ####### 
def I_withd(link_data): # integratedを読み込む想定
    n = len(link_data)
    I = np.eye(n)
    for i in range(n):
        D = link_data.loc[i, 'D']
        for j in range(n):
            if link_data.loc[j, 'O'] == D: # DnodeをOnodeとするリンクを1にしている
                I[i, j] = 1
    return(I)

###### （無向）接続行列 ###### 
Ind = I_nod(df_link_odds) 

###### （有向）接続行列 ######
Id = I_withd(df_link_integrated) 

###### 観測モデル内で使うためのデータ準備 ####### 
gate_link = [59, 61, # 1F南
            43, 47, 53, # 1Fハチ公
            35, 37] # 3F中央

# 改札リンクのうち仮想リンクに接続しないリンク
non_gate_link = [31, 33, 39, # 3F中央
                41, 45, 49, 51, 55, 57, # 1Fハチ公 link45は検討
                63, 65, 67, 69] # 1F南 

# 改札ノードのうち吸着ノードに繋がっているノード（流用可）
gate_node = [33, 36, # 1Fハチ公
            21, # 3F中央
            26, 27] # 1F南

# ホームノードの抽出
home_node = df_node[df_node['floor'] == 20.5]['nodeid'].to_list()

# 以下は観測モデルで使うため奇数を指定
home_link = [i for i in range(1, 30, 2)]
staire_link = [i for i in range(71, 100, 2)]

# 仮想リンクとの接続関係（流用可）
virtual_connect = np.array([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 35, 37, 43, 47, 53, 59, 61],
                           [106, 107, 108, 108, 109, 109, 110, 111, 112, 113, 114, 114, 115, 116, 117, 103, 103, 101, 101, 102, 104, 105]])

####### パラメタ初期設定 ####### 
x0 = np.zeros(9)
x0[0] = 0.1
x0[-1] = 0.99

####### 観測方程式 ###### 
def DDR(rssi, dens, x):
    return (10**((-rssi-50)/dens/x[0])) # x[0]は非零．RSSI0=-50は根拠なし

####### 観測モデルその1 ######
def measuring_model1(x): 

    measuring_result1 = pd.DataFrame(columns=['user_id'] + ['timestep'] + [str(i) for i in range(1, L-1)])

    for file_name in file_list:
    
        if file_name.endswith('.csv'):
            
            base_name = file_name.split('.')[0] # 数字部分のみ
            user_id = int(base_name) 
            
            file_path = os.path.join(search_folder, file_name)
            df = pd.read_csv(file_path) 
            
            # timestep数3以下の場合スキップ
            timestep_count = df['timestep'].nunique()
            if timestep_count <= 3:
                continue
            
            grouped = df.groupby('timestep') 
            df_list = [group for name, group in grouped]
            df_list = [mac_data.reset_index(drop=True) for mac_data in df_list]
            
            # res_unionの初期化．リンクL-2=117+userd_id+timestepで119（preの場合）
            res_union = np.zeros(L)

            # timestep類の用意
            df_first = df_list[0]
            df_last = df_list[-1]
            first_ts = df_first.loc[0, 'timestep']
            last_ts = df_last.loc[0, 'timestep']
            prev_ts = first_ts - 1
            timestep_range = last_ts - first_ts + 1 # このuserの合計timestep数

            # 各timestepにおける観測確率の算定
            for i in range(timestep_count): # for i in range(timestep_range)かfor i in range(first_ts, last_ts+1)
                
                df_now = df_list[i] # timestep内の個別データの集合(dataframe)

                # 1timestep内での観測尤度を入れるための配列．後でres_unionに結合．各linkの観測確率を入れるので吸収以外のリンク数117（preの場合）
                res_t = np.zeros(L-2)

                now_ts = df_now.loc[0, 'timestep'] 
                #timestep_delta = now_ts - prev_ts

                # timestep非連続と連続の場合わけ
                while (now_ts - prev_ts) > 1: # ここでwhile???? 
                    # i-1の観測結果はres_unionの一番最後の行．0, 1列はuseridとtimestepなので2列以降がリンク観測結果
                    prev_result = res_union[-1, 2:] 

                    # i-1で観測確率!=0のリンクを抽出
                    nonzero_indices = np.where(prev_result != 0)[0]     # index=linkid-1が得られる
                    prev_link = nonzero_indices + 1                     # これでprev_tsでの観測があったリンクのlinkidのリストが得られた

                    # 有効な接続リンクを抽出
                    link_candidates = set()                             # setは重複を許容しない
                    for candidate in prev_link:                         # それぞれの観測リンク（candidiateのこと，candidateはlinkid）に対して，接続するリンクのlinkidを得て，これをlink_candidatesに追加．
                        cand_column = Ind[:, candidate//2]              # candidate-1がcandidateのindex
                        cands_index = np.where(cand_column == 1)[0]     # candidateに接続するリンクcandsのindexの集合を得る．接続行列Indを参照
                        candidates = cands_index*2 + 1                  # linkidに戻す

                        for cand in candidates:
                            link_candidates.add(cand)

                    # これでlink_candidatesに観測されたリンクのlinkidが重複なく入った→リスト化→昇順並び替え（これでlinkidとindexを紐づけられる）
                    link_candidates = list(link_candidates)
                    link_candidates.sort()

                    # linkidからindexを得る（index = linkid-1）
                    cand_index = [int(x)-1 for x in link_candidates] 

                    # 判定用．連続の場合の処理をまねる
                    judge = np.zeros(L-2) 
                    index = np.array([i for i in range(L-2)]) 

                    mask = np.isin(index, cand_index)
                    judge[index[mask]] = 1
                    judge_sum = np.sum(judge)

                    if judge_sum == 0: # 接続行列がないリンクはないはずなので，実行されないはず
                        print('error!!')
                        link_probability = judge
                    if judge_sum != 0:
                        link_probability = judge/judge_sum
                    
                    # t, userid追加してres_unionの一番下に追加する．以下連続の場合と同じ
                    time_step_array = np.array([prev_ts+1])
                    t_result = np.concatenate((time_step_array, link_probability))

                    user_id_array = np.array([user_id])
                    t_result_withid = np.concatenate((user_id_array, t_result))
                    #print(f'補正あり{prev_ts+1}')

                    res_union = np.vstack((res_union, t_result_withid))

                    prev_ts = prev_ts + 1 # 処理終わったらスタンプ更新

                # while抜けたらこっちに移る，という処理ならifインデントを外して良さそう？
                #if timestep_delta == 1: 
                for j in range(len(df_now)): 
                    rssi = df_now.loc[j, 'RSSI'] 
                    dens = df_now.loc[j, 'dens']
                    ID = df_now.loc[j, 'ID']

                    ble_index_list = df_ble[df_ble['ID']==ID].index.tolist()    # index()はindexオブジェクトを返す！
                    ble_index = ble_index_list[0]                               # ble beaconのindex

                    # ddrの判定を一気に実行
                    ddr = DDR(rssi, dens, x)

                    condition = d_array[:, ble_index] <= ddr                    # d_arrayのbleindexの列の中でddrよりも値が小さいという条件（つまりddr内）
                    selected_indices = np.where(condition)                      # これがddr内認定されたlinkのindex．実際のlinkidはindex*2+1，全体で見た時のindexはindex*2
                    #print(type(selected_indices)) # tuple                      # 空のtupleはlen1になる．tuple[0]でちゃんといけるらしい(!!!!!?)
                        
                    # ddr以下を満たすlinkがなかった場合，最もdが小さいlinkを入れてselected_indicesとして更新
                    if len(selected_indices[0]) == 0: 
                        min_value = np.min(d_array[:, ble_index])
                        selected_indices = np.where(d_array[:, ble_index] == min_value)[0]  # 最小値と一致するすべての要素のインデックスを取得→where!!
                        
                    selected_linkid = [i*2 for i in selected_indices] # 該当するlinkの全体でのindex
                        
                    # judge配列の初期化．後でres_unionとくっつけるがjudgeはlinkの情報のみ．その後でuser_idとtimestepが足される
                    judge = np.zeros(L-2)
                    judge[selected_linkid] = 1 
                        
                    res_t = np.vstack((res_t, judge))
                    
                # このtimestepでの全データを捜索終了．res_tには各データでDDR内判定されたリンクのところが1になっている
                judge_counts = np.sum(res_t, axis=0) 
                judge_sum = np.sum(judge_counts)

                if judge_sum == 0:
                    print('error!!') # judge_counts=0にはならないはず
                    link_probability = judge_counts
                if judge_sum != 0:
                    link_probability = judge_counts/judge_sum # 上手い方法わからずとりあえず等配分．numpy同士なので割れている．listだと割れない

                # 要素数は1．timestepの値をlink_probabilityの先頭に入れる．そのためにはnumpy配列化してから挿入する必要がある
                time_step_array = np.array([now_ts])

                # time_step，useridを記録
                t_result = np.concatenate((time_step_array, link_probability))
                user_id_array = np.array([user_id])
                t_result_withid = np.concatenate((user_id_array, t_result))
                #print(f'補正なし{now_ts}') 

                res_union = np.vstack((res_union, t_result_withid))

                prev_ts = now_ts # 処理終わったらスタンプ更新，絶対に次のdfに進む

            # ここまでで全timestepに対して結果が出て，res_unionに入ってる．初期設定で入れておいた0の列を消去
            res_union = np.delete(res_union, 0, axis=0)
            
            # res_unionをdataframe化
            df_result = pd.DataFrame(data=res_union)

            # columnsを設定．第0列目はuser_id, 1列目はtなので，link1~117の尤度の情報が入ってるのはindex2~118列
            df_result.columns = ['user_id'] + ['timestep'] + [str(i) for i in range(1, L-1)] 

            # 確率最大リンクが複数ある場合，idxmaxメソッドは見つかった最初のindexを返す．linkの設定で階段は後ろの方に設定しておいたので，コンコースかホーム上のリンクが優先される．よって階段とホームの尤度が最大であったときもホームを採用するから問題なし
            o_link = df_result.iloc[0, 2:L].idxmax() 
            #d_link = df_result.iloc[timestep_count-1, 2:L].idxmax()
            d_link = df_result.iloc[-1, 2:L].idxmax()

            o_link = int(o_link)
            d_link = int(d_link)
            
            if (o_link in staire_link) or (d_link in staire_link):  # o_linkあるいはd_linkが階段リンクだった場合，ループを飛ばす
                continue
            if d_link in non_gate_link:                             # dがnon_gate_linkの場合落とす→これの条件がきつすぎる
                continue
            if (o_link in home_link) and (d_link in home_link):     # odともにhomeのケースは厄介そうなので落とす
                continue
            if d_link in home_link:                                 # d_linkがhome_linkを落とす
                continue

            measuring_result1 = pd.concat([measuring_result1, df_result], ignore_index=True) # ここまでであるuserに対する処理が終了，全体のdataframeに情報追加 # count += 1 

    return measuring_result1

####### 観測モデルその2 ###### # 有向グラフへの対応と仮想リンクの情報
def measuring_model2(x):
    measuring_result1 = measuring_model1(x) # 第一段階のアウトプット．列カラムはuserid, timestep, link1~117の観測確率

    # measuring_result2の初期化，最初はNaN
    measuring_result2 = pd.DataFrame(columns=['user_id'] + ['timestep'] + [str(i) for i in range(1, L-1)] + ['absorption']) 

    # user_idでgroupわけしdataframeをリスト化（いつも通り）
    grouped = measuring_result1.groupby('user_id')
    df_list = [group for name, group in grouped]
    df_list = [data.reset_index(drop = True) for data in df_list]
    
    for i in range(len(df_list)): # 処理は全部個人ごとに行ってしまい，後でdataframeを再度まとめる．
        
        df = df_list[i]
        n = len(df) # 最後の行はn-1

        last_ts = df.loc[n-1, 'timestep'] # -1だとダメだったがn-1にしたらOKになった
        last_ts = int(last_ts)
        userid = df.loc[0, 'user_id']

        # d_linkを取得（一番最後のデータから）．列名なのでintに変換する
        d_link = df.iloc[-1, 2:L].idxmax()
        d_link = int(d_link) # linkid

        # 仮想リンク用の行を下に追加．結構面倒
        new_row = pd.Series([0] * len(df.columns), index=df.columns)
        df = df.append(new_row, ignore_index=True) # 追加してるのがseriesだったのでconcatだとおかしなことになってた模様
        #df = pd.concat([df, new_row], ignore_index=True)
        df.loc[n, 'user_id'] = userid 
        df.loc[n, 'timestep'] = last_ts+1 

        # d_linkに対応する仮想リンクに確率1を与える
        is_equal = (virtual_connect[0] == d_link) # bool配列
        equal_columns = np.where(is_equal)[0] # これはリストで返ってくる
        column = equal_columns[0] # intに
        virtual_link = virtual_connect[1, column] # これで対応する仮想リンクのlinkidが得られるはず(int)### ここまでは問題ないはず
        
        # virtual_linkはlinkidなのでcolumnsに変換するには+1．measuring_result1は0: user_id, 1: timestep, 2~118がlink1~117なので
        column_num = virtual_link + 1 
        df.iloc[n, column_num] = 1 # 対応する仮想リンクの列の値を1に
        
        #### 以上でリンクに時系列の観測確率が入り，吸収状態の情報も入った．あとはお尻から次のtimestep（timestepは1からじゃない）での観測確率が最大のリンクのo_nodeに近い方をとる．# こうすることで回遊とか彷徨い行動をカットしないで済む（それだけのためである）
        
        for j in range(n-1, -1, -1): # 最終観測リンク(index = n-1，index nが仮想リンクのため)から先頭まで．tsがindexになってる．last_tsをintにするの忘れない（前）# 次のtimestep(ts+1)で最も観測確率の高かったリンクのlinkidを取得
            next_maxlink = df.iloc[j+1, 2:L].idxmax() 
            next_maxlink = int(next_maxlink) # int変換

            # df_link_intregratedを参照してmaxlinkのo_nodeを取得
            maxlink_onode = df_link_integrated[df_link_integrated['linkid'] == next_maxlink]['O'].iloc[0]

            # link1から99の奇数リンク（仮想リンクを除くので100まで）
            for k in range(1, 100, 2): # df_link_integratedを参照してlink kとlink k+1のd_nodeを取得し，o_nodeとの距離を比較
                # k(1, 3, 5, 7, )がlinkidになっているのでdf_integratedにおけるindexはk-1(0, 2, 4,,)
                klink_dnode = df_link_integrated.loc[k-1, 'D'] #Dnodeのnodeidが得られる
                k1link_dnode = df_link_integrated.loc[k, 'D'] # kと逆方向の偶数リンクのDnodeのnodeid

                d_k = d_node_array[maxlink_onode - 1, klink_dnode - 1]
                d_k1 = d_node_array[maxlink_onode - 1, k1link_dnode - 1]
                if d_k <= d_k1: 
                    continue
                if d_k > d_k1: # ならdf[k]とdf[k+1]を入れ替える．kはlinkidで今dfはuser_idとtimestepを持つからlink kのindexはk+1(link1はindex2, ,,,)，link k+1はindexk+2
                    df.iloc[j, k+2] = df.iloc[j, k+1] # k+1にkの観測確率をあげる
                    df.iloc[j, k+1] = 0 # kの方は0になる
                        
        # 最後にabsorption列追加 
        df['absorption'] = 0
        if d_link in gate_link:
            df['absorption'] = 118            
        if d_link in home_link:
            df['absorption'] = 119

        # ここまででこのユーザに対して補正が完了．付け足していく
        #measuring_result2 = measuring_result2.append(df)
        measuring_result2 = pd.concat([measuring_result2, df], ignore_index = True)
    
    # 最後にindexをリセット
    measuring_result2.reset_index(drop = True)
    measuring_result2 = measuring_result2.fillna(0)
        
    # アウトプットは，measuring_result1に各ユーザの仮想timestepが増えたもの & 基本的な形は同じ．これを尤度関数の時に呼び出す．
    return(measuring_result2)


###### 経路選択モデル ####### 

###### 初期設定 ######
ddata = [118] # 吸収リンクのlinkid # 118に限定
# L = len(df_link_integrated) # 吸収リンク含めた全リンクの数（最初に定義済み）
D = len(ddata) # 吸着リンク数＝吸着ノード数(absorption)=2
V0 = np.full((L, D), -1) # exp(0)=1よりexp(-1)は小さくなるから?
z0 = np.exp(V0)

V = V0
z = z0
beta = x0[-1] # dRLなのでbeta推定．xの最後の要素をbetaとする
#beta = 0.5 # dRLなのでbeta推定．xの最後の要素をbetaとする

###### 効用関数 ######
def linkv(x): 
    # 効用関数は検討中．x[0]は観測モデルのパラメタなのでx[1]から使う
    vinst = df_link_integrated['length']/10 * x[1] + df_link_integrated['width_home']/10 * x[2] + df_link_integrated['width_staire']/10 * x[3] + df_link_integrated['staire_with_esc'] * x[4] + df_link_integrated['hachi_dum'] * x[5] + df_link_integrated['minami_dum'] * x[6] + df_link_integrated['chuo_dum'] * x[7] 
    #vinst = df_link_integrated['length']/10 * x[0] + df_link_integrated['width_home']/10 * x[1] + df_link_integrated['width_staire']/10 * x[2] + df_link_integrated['staire_with_esc'] * x[3] # + df_link_integrated['hachi_dum'] * x[5] + df_link_integrated['minami_dum'] * x[6] + df_link_integrated['chuo_dum'] * x[7] 
    #vinst = df_link_integrated['length']/10 * x[0]
    ############
    #####ここでホーム→改札の人ならhome*lengthの負効用が大きい（つまり電車降りたらホームあんま歩きたくない），改札→ホームの人ならgate*lengthの負効用が大きい（つまり改札入ったらすぐホームに行きたい），という差が出るはず
    # なのでlinkvを書き換えた方が良い．#######
    #############
    vinst = np.exp(vinst) # ここで指数関数化
    return vinst # 出力形式はseries

###### 即時効用行列 ######
def Mset(x): 
    cost = linkv(x)
    cost = pd.concat([cost]*L, axis=1)
    cost = cost.T
    # DataFrameをNumPy配列に変換
    cost_numpy = cost.values
    return cost_numpy # 出力はdataframe

###### 価値関数 ######
def newV(x): 
    V = np.zeros((L, D))
    z = np.exp(V)
    
    for d in range(D): # dごとに処理
        d_linkid = ddata[d] 
        d_index = df_link_integrated[df_link_integrated['linkid'] == d_linkid].index # ちゃんと出てる（list）

        z[d_index, d] = 1  # 目的地がdの時，dでの期待効用Vd(d)=0より
        
        M = np.zeros((L, L))
        B = np.zeros((L, 1))
        B[d_index, 0] = 1
                
        # Mをdごとに更新
        for k in range(L):
            for a in range(L): 
                Ika = Id[k, a] # kもaもindexなのでd_linkidで対応．Idは普通の接続行列でいいのでは
                if Ika == 1: # 接続してなければ無関係
                    if a == d_index:
                        M[k, a] = 1  # 吸収リンク効用はexp(0)=1
                    else:
                        M[k, a] = Mset(x)[k, a] 

        dL = 100
        zd = z[:, d].reshape(L, 1) # zのd列目のみ取り出して列ベクトル化（reshapeは保険）

        # z求解
        while dL >= 0.01:
            zdd = zd.copy()
            Xz = zd**beta
            zd = np.dot(M, Xz) + B # 定義通りのはず（L*1のmatrix）@でも変わらない
            #print(zd) #zdがnanになってループ終了している
            dL = np.sum(np.abs(zdd - zd)) # np.absが各行の差分の配列．np.sumでその和をとっている．

        # z更新
        z[:, d] = zd[:, 0] # 収束させた後でzのd列目を更新(zdは1列しかないのでz[:, d] = zdでも良さそう)
        # zdの最終行が1になる．なぜだ，，，
        # zdとnp.log(zd)が119*1の配列である一方，V[:, d]は119の1次元numpy配列

        # V更新
        zd = np.where(zd == 0, 1, zd)  # 回避のための操作log(0)
        one_dim_array = np.log(zd).ravel() # 1次元配列に
        V[:, d] = one_dim_array # reshapeしなくてもOKそう
    
    return z # 119*2のnumpy配列を返す


###### 対数尤度関数 ###### 
def loglikelihood(x):
    LL = 0

    # 観測モデルとの融合
    print(f'パラメタ{x}で観測モデルを回します')
    measuring_result2 = measuring_model2(x)
    print('観測モデルは回りました')

    # 観測結果をdで分割
    grouped = measuring_result2.groupby('absorption')
    df_list = [group for name, group in grouped]
    df_list = [data.reset_index(drop=True) for data in df_list]    

    for d in range(D):
        #print(f"ll関数内でd={d}を見ている")
        d_linkid = ddata[d] # ddata=[改札吸収linkid，ホーム吸収linkid]
        d_index = df_link_integrated[df_link_integrated['linkid'] == d_linkid].index
        d_index = d_index.tolist()
        d_index = d_index[0] # indexはちゃんと出ている！

        # 価値関数
        z = np.exp(beta*V[:, d]).reshape(L, 1) # V所与として処理．newV内でのz（D列）とは違うので注意
        # ZD = np.array([z]*L) # コピーするには[]をつけないといけない！ZDはL*L###### 超注意！！！arrayをコピーすると3次元になってしまう．横に付け足すのはtile!!########
        ZD = np.tile(z, (1, L))
        ZD = ZD.T

        # 瞬間効用        
        M = Id * Mset(x) #*で要素ごとの積
        M[:, d_index] = 1 # np.exp(0) # 吸収リンクへ遷移する際の瞬間効用は1

        # 選択確率
        Mz = (M @ z != 0) * (M @ z) + (M @ z == 0)  # MZ = np.array([Mz]*L) これ3次元になるのでnp.tileを用いる！よ
        MZ = np.tile(Mz, (1, L))  # MZ = MZ.T # MZの方は転置しない，MZから先に掛けるので．順番の都合）

        p = (M * ZD) / MZ # *で要素ごとの積（Rと同じ）
        p = (p == 0) * 1 + (p != 0) * p  # これでp[k, a]にはk→aの遷移確率が入ってる（多分numpy）

        # ここでdを目的地とするユーザのデータを読み込む
        df = df_list[d]
        
        grouped2 = df.groupby('user_id')
        df_list2 = [group for name, group in grouped2] # df_list2の長さが目的地をdとするuserの数に等しい
        df_list2 = [data2.reset_index(drop=True) for data2 in df_list2]

        Ld = 0 # dのための対数尤度
        for i in range(len(df_list2)): #iは個人
            count = 0
            df_indivi = df_list2[i] # 各自のデータ # 列カラムはuser_id, timestep, link1~117（columnの観測確率，absoptionの120列

            data_subset = df_indivi.iloc[:, 2:L]
            q = data_subset.to_numpy() # q[t, k]が，個人のtimestep t番目のlink k+1の観測確率に対応
            
            li = 1 # 個人ごとの初期の尤度

            # timestepごとにステップ毎に尤度出して行く
            for t in range(2, len(q)): 
            #そもそもtimestep1は前を参照しようがないのでtimestep2から処理．しかしtimestep2の場合，1がうまく取れてないと0になって後ろが死ぬ．別に頭がそれほど重要なわけでもないので，timestep3からやってみる．t:3~len(q)．timestep3からやっても実際にはtiemstep2の情報を使ってるので十分出だしの挙動は考慮できそう
                # 時刻tにおいてq != 0のlinkidを取得→measured_link
                measured_links = np.where(q[t] != 0)[0] # list形式．index
                measured_links = measured_links + 1 # linkid

                # 時刻tの時の観測尤度の初期化
                lt = 0

                # 時刻tでの各観測リンクに対しての操作
                for measured_link in measured_links: # measured_linkはlinkid
                    l_for_each = 0
                    measured_column = Id[:, measured_link-1] # linkid-1なのでlink index．measured_linkをdとするリンクのところが1になってる
                    #print(f'measured_linkの接続行列{measured_column}')
                    pre_link_list_index = np.where(measured_column == 1)[0] # measured_linkidに接続するリンクpre_link集合を得る．接続行列Idを参照
                    pre_link_list = pre_link_list_index + 1 # linkid
                    #print(f'今のlinkidは{measured_link}で，接続してるはずのlinkのlinkidは{pre_link_list}') # 自分は自分に接続してるので，
                    for pre_link in pre_link_list:
                        # pre_link→measured_linkへの遷移確率：p[pre_link-1, measured_link-1]，pre_linkのt-1での観測確率：q[t-1, pre_link-1]，measured_linkの時刻tでの観測確率：q[t, measured_link-1]
                        l_for_each = l_for_each + q[t-1, pre_link-1] * p[pre_link-1, measured_link-1] * q[t, measured_link-1]
                        #print(q[t-1, pre_link-1], p[pre_link-1, measured_link-1], q[t, measured_link-1])
                        #print(f'{l_for_each})
                        # 各リンクの尤度が足され，l_for_eachが完成
                        #print(f'毎回のl_for_each{l_for_each}')
                    
                    lt = lt + l_for_each # 各measured_linkのlink尤度を積み上げる
                    # これで時刻tの尤度ltがでた．後はLiにltを順次かけていく（対数尤度なら，足していく）
                
                #print(f'時刻{t}で尤度ltは{lt}') # 時刻1でltが0になっている→残り全部0になってしまう，ltが0ならliが0になってlliが0になってしまう．
                li = li * lt
                #print(f'時刻{t}で尤度liは{li}')
                        
            # これで個人iの尤度liがでた．これを対数化            
            # lliが0ならcontinue→次のi(user)に（log zero回避）
            if li == 0:
                continue
                
            else:
                lli = math.log(li)
                Ld += math.log(li) 
                count += 1

        LL += Ld
        print(f"link{d}の操作終わり，計算したuserは{count}人で，今尤度は{LL}です")

    print('以上で一回loglikelihoodが回りました')
    return -LL

###### 推定部分 ###### 
dL = 100
n = 0

x_init = x0
#bounds = [(0, None), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10), (0, 1)]

while dL >= 5:
    print(f"<<<<<<<<<<<<<<<<now at {n}th loop in while cond>>>>>>>>>>>>>>>>>>")
    n += 1
    x = x0
    V = V0
    z = z0 # 価値関数固定

    # 構造推定part1：対数尤度関数最大化
    res = minimize(loglikelihood, x, method='BFGS', options={"maxiter":10,"return_all":True}) 
    
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
print("時間割引率 = ", x0[-1])
print("t値 = ", tval)