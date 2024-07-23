import pandas as pd
import os
import numpy as np
import random

base_path = '/Users/takahiromatsunaga/res2023/PPcameraTG/PP/PP_webq'
# df_mezzo_link = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/map/shibuya_mezzo_link.csv'))
# df_mezzo_node = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/map/shibuya_mezzo_node.csv'))
# df_mezzo_nw = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/map/shibuya_mezzo_nodebased_network.csv'))
# df_gate = pd.read_csv(os.path.join(base_path, 'PP_webq/gate_MNL/gate.csv'))
# df_destination = pd.read_csv(os.path.join(base_path, 'PP_webq/gate_MNL/destination.csv'))
# df_micro_node = pd.read_csv(os.path.join(base_path, '../../shibuya_nw/shibuya_stanw/micro_node_4326.csv'))

timing = 'post' 
#timing = 'underconst'
#timing = 'prior'
#sotouchi = 'soto'
#sotouchi = 'uchi'
timerestriction = 15
point_gate = 3
point_amount = 0
signed_point_percent = 0

df_link = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_link_post_all_newminami.csv')
df_node = pd.read_csv('/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/stanw_node_all_newminami.csv')
df_odmat = pd.read_csv(os.path.join(base_path, f'gate_MNL/gate_assignment_0610/gate_assign_gate{point_gate}_{point_amount}point_{signed_point_percent}percent_abs.csv')) ## odid, o, d, demands,  これ改札モデルから用意する
## absで出口の別を決めておくか，，，

assumed_x = np.array((-1, 0.85)) ## 例えばlengthとbetaだけなら
#staire_index_dif = 100 ## ありえない値を入れておくことで間違って回るのを回避
ddata = []
#lastdata = []

if timing == 'post':
    #staire_index_dif = 20
    #ddata = [66, 67, 68]
    ddata = [66, 67, 68, 69, 70, 71] # 配分なので全改札を使う
    #lastdata = [25, 17, 18, 22, 23] # 最後中央改札（ビーコン16）とハチ公ビーコン20, 21の近くのリンクをこの順で入れる
    #yamate_key = list(range(7, 14)) # [7, 8, 9, 10, 11, 12, 13]
    #first_staire_id = 26

L = len(df_link)
T = timerestriction + 1 # 20ステップ以内ならT=21
D = len(ddata)
OD = len(df_odmat)

###### リンク接続行列＆プリズム準備 #######
def Imaking(link_data): 
    n = len(link_data)
    I = np.eye(n) # 滞在もUターンもOK
    for i in range(n):
        #if (i < (first_staire_id - 1)) or (i >=  (ddata[0] - 1)): 
        O = link_data.loc[i, 'o'] 
        D = link_data.loc[i, 'd'] 
        for j in range(n):
            if ((link_data.loc[j, 'o'] == O) or (link_data.loc[j, 'o'] == D)) or (link_data.loc[j, 'd'] == O) or (link_data.loc[j, 'd'] == D): 
                I[i, j] = 1
        # if (first_staire_id - 1) <= i < (ddata[0] - 1): ### 階段リンクの場合
        #     O = link_data.loc[i, 'o'] # onode
        #     D = link_data.loc[i, 'd'] # dnode
        #     for j in range(n):
        #         if link_data.loc[j, 'o'] == D: # DをOnodeとするリンクを1にしている
        #             I[i, j] = 1
        #         if link_data.loc[j, 'o'] == D or link_data.loc[j, 'd'] == D: # odいずれかがDと一致するリンク[からの]遷移は0
        #             I[j, i] = 0
        #         if link_data.loc[j, 'o'] == O or link_data.loc[j, 'd'] == O: # odいずれかがOと一致するリンクへの遷移は0
        #             I[i, j] = 0
    return(I)

I = Imaking(df_link)
def TSNW(ODlist): ### 引数ちゃんと考える必要！ODのリストという点ではdf_odmatが優れている
    Ilist = []

    Itlist = np.zeros((T, L, L)) ## これは各状態に対して与えるので遷移回数T-1よりひとつ多くてT成分（timestep数がTなら状態数はT）
    II = np.copy(I)
    Itlist[0, :, :] = np.eye(L) 

    for ts in range(1, T): 
        Itlist[ts, :, :] = II 
        II = np.dot(II, I)
        II = np.where(II > 0, 1, 0) 

    Ittlist = np.zeros((T, L, L))
    for ts in range(T): 
        Ittlist[ts, :, :] = np.transpose(Itlist[T - ts - 1, :, :])

    OD = len(ODlist)
    for od in range(OD):   
        ao = ODlist.loc[od, 'dcarnode'] - 1  
        aabs = ODlist.loc[od, 'abs'] - 1  

        Id = np.zeros((T-1, L, L)) # Id[t]は時刻tの時の利用可能遷移k→aを示す．最後のt=T-1のとき不要なので要素数はt=0~T-2のT-1個

        for ts in range(T-1): # Tまでに吸収されるのでdはaabs．遷移回数はT-1なのでこれでOK
            if ts == 0:
                Id[ts, ao, :] = I[ao, :] 
                continue
            
            #alist = np.where((Itlist[ts + 1, ao, :] == 1) * (Ittlist[ts + 1, aabs, :] == 1) == 1)[0] ## ts=1（つまり第二回目の状態）のとき3番目の状態を見ている→OK
            #alist = np.where(Itlist[ts + 1, ao, :] == Ittlist[ts + 1, aabs, :])[0] ## ts=1（つまり第二回目の状態）のとき3番目の状態を見ている→OK
            alist = np.where(Ittlist[ts + 1, aabs, :] == 1)[0]
            # if od == 7:
            #         print(f't={ts}でalistは{alist}')

            for a in alist:
                #if Itlist[ts + 1, ao, a] == Ittlist[ts + 1, aabs, a]: # always True


                if Ittlist[ts + 1, aabs, a] == 1:
                    #if od == 7:
                    #    print(f't={ts}でalistは{alist}')

                    klist = np.where(I[:, a] == 1)[0]
                    #if od == 7:
                        #print(f'od{od}t{ts}, aは{a}klist{klist},aoは{ao}, aabsは{aabs}')
                    
                    for k in klist:
                        if len(np.where(Id[ts - 1, :, k] == 1)[0]) != 0:
                            Id[ts, k, a] = 1 
                            #if od == 7:
                                #print(f'od{od}, t{ts}, k{k}, a{a}, Id{Id[ts, k, a]}')
        Ilist.append(Id)
    
    return Ilist

Ilist = TSNW(df_odmat)

####### 経路選択モデル ######
def Mset(x): 
    inst = np.zeros((T-1, L, L))
    for t in range(T-1):
        inst_t = np.exp(df_link['length']/10 * x[0]) # + Probs_T[:, t] * x[1])    # + df_link['staire'] * x[1]) # + df_link_integrated['staire_with_esc'] * x[3])
        inst_t = pd.concat([inst_t]*L, axis=1) 
        inst_t = inst_t.T
        inst_t_numpy = inst_t.values # DataFrameをNumPy配列に変換
        inst[t, :, :] = inst_t_numpy
    return inst


def newPall(x):
    Pall = np.zeros((OD, T-1, L, L)) # 個人ごと，時刻ごとの各リンク間遷移確率行列 遷移の回数なのでT-1．今回Tを状態の数としているので遷移数はT-1になる
    beta = x[-1]

    for od in range(OD):
        Id = Ilist[od] # すでに個人のプリズムはIlistで用意してある

        M = np.zeros((T-1, L, L)) # Tが状態数なので遷移数はT-1
        for ts in range(T-1):
            Mts = Id[ts, :, :] * Mset(x)[ts, :, :] # 内生性考慮してないのでMsetは定常
            M[ts, :, :] = Mts
            
        z = np.ones((T, L))
        for t in range(T-1, 0, -1):
            zii = M[t-1, :, :] * (z[t, :] ** beta) 
            zi = zii.sum(axis = 1)
            z[t-1, :] = (zi==0)*1 + (zi!=0)*zi

        for t in range(T-1): 
            for k in range(L):
                for a in range(L):
                    if M[t, k, a] == 0: # 接続条件を満たせなかった観測は排除（logzero回避）
                        continue # ここがbreakになってたのが癌だった         
                    Pall[od, t, k, a] += np.exp(np.log(M[t, k, a]) + beta * np.log(z[t+1, a]) - np.log(z[t, k])) 
    return Pall 


###### 配分関数 ######
def assignment(x): 
    Pall = newPall(x)
    res_all = np.zeros(4) # columns = ['userid', 't', 'k', 'd']
    for od in range(OD): # ODのペア数
        ao = df_odmat.loc[od, 'dcarnode'] - 1 ### odmat_initで
        aabs = df_odmat.loc[od, 'abs'] - 1 # 吸収リンク # 時刻Tまでに吸収リンクに到達するという設定なのでここはabsを入れてて良いと思う
        oddemands = df_odmat.loc[od, 'counts']

        # 累積確率行列
        Pi = np.copy(Pall[od, :, :, :])

        for t in range(T-1): 
            for k in range(L):
                if k == 0:
                    Pi[t, :, k] = Pi[t, :, k]
                else:
                    Pi[t, :, k] += Pi[t, :, k-1]

        # if timing == 'prior':
        #     o_index = od % 14 # 5 
        #     d_index = od // 14
        # else: # timing == 'post' or 'underconst':
        #     o_index = od % 7
        #     d_index = od // 7
        # # o_index = od // 5
        # # d_index = od % 5
        
        ## 発生交通量
        # Nod = int(demand_mat[o_index, d_index]) ## これで作成したOD表を参照して需要を出してる
        # print(f'od={od}でoindex={o_index}, dindex = {d_index}, Nod={Nod}')
        # if Nod == 0:
        #     continue # odペアが存在しないならパス

        for nod in range(oddemands): # 個人ごとに経路配分
            res_indivi = np.zeros((T, 4))
            res_indivi[:, 0] = nod          # nod(userid)
            res_indivi[:, 1] = np.arange(T) # timestep
            res_indivi[:, 3] = aabs + 1       # 吸収リンクid

            ran_list = [random.random() for _ in range(T-1)] ## 乱数遷移回数T-1分発生

            kn = ao + 1                     # id
            k = ao                          # index
            res_indivi[0, 2] = kn
            #count = 0
            for t in range(T-1): 
                ran = ran_list[t]
                # print(f'od{od}, nod{nod}, time{t}')
                for a in range(L):
                    # print(f'od{od}, nod{nod}, time{t}, a={a}')
                    if Pi[t, k, a] > 1:
                        #print(f'dっっっっsOD{od}, 個人{nod}時間{t}で確率{Pi[t, k, a]}')
                        k = a
                        kn = k + 1
                        res_indivi[t+1, 2] = kn

                    if ran <= Pi[t, k, a]:
                        k = a
                        kn = k + 1
                        res_indivi[t+1, 2] = kn
                        break
            
            #if count == 1:
                #print(f'od{od}, nod{nod}') #, time{t}, a={a}, kn={kn}')
            
            res_all = np.vstack((res_all, res_indivi))

    res_all = np.delete(res_all, 0, axis = 0)
    df_res_all = pd.DataFrame(res_all, columns=['userid', 't', 'k', 'abs'])

    ### 完成した経路選択行動データ
    #df_res_all_filename = f'assign_res/assigned_{time_stamp}_allkai.csv'
    #df_res_all.to_csv(os.path.join(base_path, df_res_all_filename))# f'assign_res_true{}.csv')) ### これ行ける・？？
    
    #df_res_all_filename = f'ODcorrected_assign/odcorrected_assign_{time_stamp}_{sotouchi}_{timerestriction}_fsame.csv'
    #df_res_all.to_csv(os.path.join(base_path, df_res_all_filename))# f'assign_res_true{}.csv')) ### これ行ける・？？
    df_res_all.to_csv(f'/Users/takahiromatsunaga/res2023/results/0610multiscale/micro_assign_{point_gate}_{point_amount}point_{signed_point_percent}percent.csv')
    return df_res_all


######################
##### 経路配分実行 #####
######################

# demand_size = df_demand['demand'].sum()
Ilist = TSNW(df_odmat)

x = assumed_x
df_res_all = assignment(x) # 関数内でcsv出力


######################
###### 可視化準備 ######
######################

# 可視化用に，各時刻の各リンクの人数を入れる．
# timestep, i for i in range(1, 69)
# vizu_all = np.zeros(1+L) # リンク数+timestep用 ## timestepづつ足そうと思ってたけど一気にやった方が楽そうなので．
vizu_all = np.zeros((T, L+1))
vizu_all[:, 0] = np.arange(timerestriction+1) # timestepが0から始まるのは変？までも0~10の方が綺麗か．

# grouped = df_res_all.groupby('t') ## timestepで分別
# df_res_all_list = [group.reset_index(drop=True) for name, group in grouped]

## まあ上から見ていって該当するセルの値を1づつ増やす方法でいいかー
for i in range(len(df_res_all)):
    t = int(df_res_all.loc[i, 't']) # df_res_allでのtが0~10で振っているのでtは0~10．なのでそのままindexとして使える
    k = int(df_res_all.loc[i, 'k']) # kはid，indexはk-1．
    vizu_all[t, k] += 1 ## 1列目がtimestepなのでindexとlink id は同じ

df_flow_all = pd.DataFrame(vizu_all, columns = ['t'] + list(range(1, L+1)))
# df_vizu_all.to_csv(f'/home/matsunaga/res2023/data/ODcorrected_assign/vizu_{time_stamp}_{sotouchi}_{timerestriction}_fsame.csv') ## 各時刻の各リンクの滞在人数のcsv
df_flow_all.to_csv(f'/Users/takahiromatsunaga/res2023/results/0610multiscale/micro_flow_{point_gate}_{point_amount}point_{signed_point_percent}percent.csv')

### 15分間の電車到着を再現したい