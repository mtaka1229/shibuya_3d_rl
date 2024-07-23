import pandas as pd 
import networkx as nx
import os

df = pd.read_csv("/Users/takahiromatsunaga/res2023/bledata/ble_timegroup/20221218_05/alladdress/359.csv")
df_link = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_link2.csv")
df_node = pd.read_csv("/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_node2.csv")

################################################################
################################################################
#################  ダイクストラ法の準備   #########################
################################################################
################################################################

def dijkstra(graph, start, goal): 
    # 開始ノードからの距離を初期化する
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0

    # 未処理のノードを初期化する
    # unprocessed_nodes = set(graph.nodes)

    # 最尤経路を格納する辞書を初期化する
    mostlikely_paths = {}

    while start != goal: 
        # 尤度が最も大きい未処理のノードを取得する
        current_node = min((node for node in graph.nodes if node != goal), key=lambda node: distances[node])

        # 最尤経路を最尤経路辞書に追加する
        mostlikely_paths[current_node] = distances[current_node]

        # 隣接ノードを取得する
        neighbors = graph.neighbors(current_node)

        for neighbor in neighbors:
            # 開始ノードから現在のノードまでの距離「尤度」を取得する
            distance = graph[current_node][neighbor]['time_cost'] ####time_cost列がないとか言われても困る

            # 現在のノードを経由した場合の距離「尤度」を計算する
            new_distance = distances[current_node] + distance

            # より短い「尤度の高い」経路が見つかった場合、距離「尤度」を更新する
            if new_distance > distances[neighbor]:
                distances[neighbor] = new_distance

        # 現在のノードを処理済みとしてマークする
        del distances[current_node]

        # 処理済みのノードを隣接ノードから削除する
        for node, _ in graph.edges(current_node):
            del distances[node]

    # 最後にゴールノードの尤度を記録しておく
    mostlikely_paths[goal] = distances[goal]

    return mostlikely_paths

#Dijkstra準備終わり

################################################################
################################################################
############### reading and proccessing data ###################
################################################################
################################################################

# RSSIと検出半径の関数を決める必要がある
def r_rssi(rssi):
    return (-1)*rssi-56 # 根拠となる論文を示すーーーIEEEのやつの方が理想的?

# link dataにlink likelihood（link costに該当）の初期値0を割り当てる
# user(file)ごとに変える必要があるのでforループの中
# いちいちoutputする必要はないのでto_csvにはしない．よって変更はforループの外には引き継がれない
df_link['link_likelihood'] = 0

# 一番最初と一番最後だけif文で別にしてstart, goalを決めるようなコードにする
for i in range(len(df)):
    # まずは観測APの座標
    x_ap = df.loc[i, 'x']
    y_ap = df.loc[i, 'y']
    z_ap = df.loc[i, 'z']

    if i == 0: # start nodeを決める
        # link idと距離を保持する辞書を用意．全リンクとの距離を計算し，最小になるlinkのlinkidがstart nodeになる
        dict = {}
        for j in range(len(df_link)):
            x = df_link.loc[j, 'x_mid']
            y = df_link.loc[j, 'y_mid']
            z = df_link.loc[j, 'z_mid']

            # rを計算
            r = ((x - x_ap)**2 + (y - y_ap)**2 + (z - z_ap)**2)**0.5

            dict.update({df_link.loc[j, 'linkid']: r})
            
        start = min(dict, key=lambda k: dict[k]) # 正味これ何してるのか意味不明

    if i == (len(df) - 1): # goal nodeを決める
        # link idと距離を保持する辞書を用意．全リンクとの距離を計算し，最小になるlinkのlinkidがstart nodeになる
        dict = {}
        for j in range(len(df_link)):
            x = df_link.loc[j, 'x_mid']
            y = df_link.loc[j, 'y_mid']
            z = df_link.loc[j, 'z_mid']

                # rを計算
            r = ((x - x_ap)**2 + (y - y_ap)**2 + (z - z_ap)**2)**0.5

            dict.update({df_link.loc[j, 'linkid']: r})
            
        goal = min(dict, key=lambda k: dict[k]) # 正味これ何してるのか意味不明

    else:
            # RSSIを取得
        rssi = df.loc[i, 'RSSI']
            # RSSIから，基準となる半径（DDR）を取得
        ddr = r_rssi(rssi)
            # 検出されたbleapの座標と，linkの中点の座標との距離がr以下の場合，そのlinkのlink idを取得
            # 各linkの中点(x_mid, y_mid, z_mid)とx_ap, y_ap, z_apとの距離がr以下かどうかを判別
            # r以下のlinkを記録するlistを用意
        under_r = []
            
        for j in range(len(df_link)):
            x = df_link.loc[j, 'x_mid']
            y = df_link.loc[j, 'y_mid']
            z = df_link.loc[j, 'z_mid']

                # rを計算
            r = ((x - x_ap)**2 + (y - y_ap)**2 + (z - z_ap)**2)**0.5 

                # ddr以下ならunder_rのlistに追加
            if r <= ddr:
                under_r.append(df_link.loc[j, 'linkid'])
            
        # 尤度likelihoodは各linkに均等に割り当てる
        l = 1/len(under_r)

        # under_rにはddr以下のlinkのlinkidが入ってる．このデータを元に，該当linkのlink_likelihood列の値をlだけ増やす
        #for k in under_r:
            # under_r[k]がlinkidである
            # 注意：linkedidはindex+1 故にindexを指定したい場合はlinkedid-1=k-1
            #df_link.loc[under_r[k], 'link_likelihood'] += l

    # Loop through linkids in under_r and update 'link_likelihood' for each valid linkid
        for k in under_r:
            if k-1 in df_link.index: # ここもk-1とする！！！！
                df_link.loc[k-1, 'link_likelihood'] += l # k-1とする点に注意
            else:
                print(f"Warning: linkid {k} not found in df_link.")
      
#############################################################
#############################################################
#############################################################
# このforループが回れば一旦は全リンクの尤度が足し算されて，特定の個人の経路に対してlink尤度の分布が完成するはず
# このコードだと尤度が全部足し算になっているので回遊や滞在を考慮できない，，，
#############################################################
#############################################################
#############################################################

    # 最大尤度経路を決めるからODのリンクを決める必要がある
    # 後で工夫するとして，とりあえず一番最初と最後に観測された観測点からそれぞれ最も近いノードをODにする
G = nx.Graph()
nodes = df_node['nodeid'].tolist()
G.add_nodes_from(nodes)

list = []
for i in range(len(df_link)):
    list.append((df_link.loc[i, 'O'], df_link.loc[i, 'D'], df_link.loc[i, 'link_likelihood']))

G.add_weighted_edges_from(list) #od(link)の組と重み（すでにdf_linkに更新してある）を入れたlistをこの前で作っておき，()内にはそれを指定すれば良い

print(dijkstra(G, start, goal))

