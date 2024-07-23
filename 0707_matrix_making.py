
# node dataとlink dataからnode-basedなnetworkデータを作る
import pandas as pd 
import csv
import os 

base_path = '/Users/takahiromatsunaga/res2023/shibuya_nw/shibuya_stanw/ver0707'
df_link = pd.read_csv(os.path.join(base_path, 'micro_link_post.csv'))
df_node = pd.read_csv(os.path.join(base_path, 'micro_node_post.csv'))

# network data: どのノードとどのノードが繋がっているか
df_network = pd.DataFrame(columns=['k', 'a'])
for i in range(len(df_node)):
    connected_nodes = [] # このnodeに接続してるnode
    nodeid = df_node.loc[i, 'nodeid'] # このノードのnodeid

    temp_nodes = [] # 今見ているノードに隣接するノードがtemp_nodesに入る

    # 条件に基づくフィルタリングとIDの追加
    if not df_link[df_link['o'] == nodeid].empty:
        temp_nodes.extend(df_link[df_link['o'] == nodeid]['d'].tolist())

    if not df_link[df_link['d'] == nodeid].empty:
        temp_nodes.extend(df_link[df_link['d'] == nodeid]['o'].tolist())

    # if not df_link[df_link['onode'] == dnode].empty:
    #     temp_links.extend(df_link[df_link['onode'] == dnode]['id'].tolist())

    # if not df_link[df_link['dnode'] == dnode].empty:
    #     temp_links.extend(df_link[df_link['dnode'] == dnode]['id'].tolist())

    temp_nodes =  list(set(temp_nodes)) 

    # Remove the linkid if it's in the temp_links list
    temp_nodes = [node for node in temp_nodes if node != nodeid] # 今のnodeidと同じやつは意味がないのでとる（o→oになって意味不明）
    connected_nodes.extend(temp_nodes)

    for connected_node in connected_nodes:
        df_network = df_network.append({'k': nodeid, 'a': connected_node}, ignore_index=True)

df_network.to_csv(os.path.join(base_path, 'nodebased_matrix_post.csv')) 