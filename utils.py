import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_to_ancestor_graph(G):
    '''Converts a (parent) tree to a graph with edges for all ancestor relations in the tree.'''
    G_anc = nx.DiGraph()
    for node in G.nodes():
        for anc in nx.ancestors(G, node):
            G_anc.add_edge(anc, node)
    return G_anc


def dataframe_to_ancestor_graph(df):
    forest = []
    for group in tqdm(list(set(df.group))):
        forest.append(group_to_ancestor_graph(df, group))
    return pd.concat(forest, ignore_index=True)


def group_to_ancestor_graph(df, g):
    df_tree=df[df.group==g]
    graph = nx.DiGraph()
    parents=df_tree['parent'].tolist()
    children=df_tree['child'].tolist()
    nodes = set(parents + children)
    for node in nodes:
        graph.add_node(node)
    for i in range(len(parents)):
        graph.add_edge(parents[i], children[i])

    T = convert_to_ancestor_graph(graph)

    df = nx.to_pandas_edgelist(T)
    df['group']=g
    df.columns=['parent','child','group']
    df['compare']=df['parent']+df['child']+df['group'].astype(str)

    return df
    

def maximum_likelihood(df_t, group):
    def weight_filter(G):
        edges_to_remove = []

        for u, v, data in G.edges(data=True):
            if data['weight'] <= np.log(0.5):
                edges_to_remove.append((u, v))

        G.remove_edges_from(edges_to_remove)
        return G
   
    return post_process_predictions(df_t, group, weight_filter)


def majority_voting(df_t, group, num_candidate=5):
    threshold = int(num_candidate / 2) + 1

    def weight_filter(G):
        edges_to_remove = []

        for u, v, data in G.edges(data=True):
            if data['weight'] < threshold:
                edges_to_remove.append((u, v))

        G.remove_edges_from(edges_to_remove)
        return G
   
    return post_process_predictions(df_t, group, weight_filter)


def maximum_absorbance(df_t, group):
    return post_process_predictions(df_t, group, nx.maximum_spanning_arborescence)


def maximum_branching(df_t, group):
    return post_process_predictions(df_t, group, nx.maximum_branching)


def post_process_predictions(df_t, group, process_func):
    df_tree=df_t[df_t.group==group]
    graph = nx.DiGraph()
    parents=df_tree['parent'].tolist()
    children=df_tree['child'].tolist()
    probabilities=df_tree['predict'].tolist()
    nodes = set(parents + children)
    for node in nodes:
        graph.add_node(node)
    for i in range(len(parents)):
        graph.add_edge(parents[i], children[i], weight=probabilities[i])
    
    T = process_func(graph)
#     T = convert_to_ancestor_graph(T)

    # convert back to pandas dataframe
    
    df = nx.to_pandas_edgelist(T)
    df=df[['source','target']]
    df['group'] = group
    df.columns=['parent','child','group']
    df['compare']=df['parent']+df['child']+df['group'].astype(str)
    
    return df



def evaluate_groups(df_actual, df_pred):
    recall = []
    precision = []
    f1 = []

    for group in tqdm(list(set(df_actual.group))):
        group_actual = df_actual[df_actual.group == group]
        group_pred = df_pred[df_pred.group == group]
        group_common = pd.merge(group_actual, group_pred, on=['compare'], how='inner')

        group_recall = len(group_common) / len(group_actual) if len(group_actual) > 0 else 0
        group_precision = len(group_common) / len(group_pred) if len(group_pred) > 0 else 0

        if group_recall + group_precision == 0:
            group_f1 = 0
        else:
            group_f1 = 2 * (group_precision * group_recall) / (group_precision + group_recall)
        
        recall.append(group_recall)
        precision.append(group_precision)
        f1.append(group_f1)

    return np.mean(recall), np.mean(precision), np.mean(f1)



def violation_val(res):
    n_root=[]
    no_root=[]
    mul_parents=[]

    for g in list(set(res.group)):
        g_net=res[res.group==g]

        G = nx.DiGraph()
        G.add_edges_from(g_net[['parent','child']].values)
        roots=  [n for n,d in G.in_degree() if d==0]
        mul_parents.append([len(list(set(g_net[g_net.child==n].parent))) for n in set(g_net.child)])
        n_root.append(len(roots))
        no_root.append(1 if len(roots)==0 else 0)
    avg_p=[np.mean(l) for l in mul_parents ]
    count_mu_p=[len([i for i in l if i>1]) for l in mul_parents]
    per_mu_p=[len([i for i in l if i>1])/len(l) for l in mul_parents]
    df_viol=pd.DataFrame(list(zip(n_root,no_root,avg_p,count_mu_p,per_mu_p)),columns=['num_root','no_root','avg_parent','count_mult_parent','perc_mult_parent'])
    return df_viol

def avg_violation_val(dict_res):
    res={'header':'count of roots'+"&"+'count of no root groups'+"&"+'avg number of parent'+'&'+'avg num of nodes with multiple parents'+'&'+'% nodes with multiple parents'}
    for key, df in dict_res.items():
        avg_num_root=round(np.mean(df.num_root),2)
        avg_no_root= round(np.mean(df.no_root),2)
        avg_parent=round(np.mean(df.avg_parent),2)
        avg_count_mu_p=round(np.mean(df.count_mult_parent),2)
        avg_per_mul_parent=round(np.mean(df.perc_mult_parent)*100,2)
        res[key]=str(avg_num_root)+"&"+str(avg_no_root)+"&"+str(avg_parent)+'&'+str(avg_count_mu_p)+'&'+str(avg_per_mul_parent)

    return res