# In-Network Aggregation: Network Build Source Code
# ----------------------------------------------------------------------------------------------------------------------

# Libraries:
# ----------
from mininet.cli import CLI
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
import os
import sys
import argparse
from glob import glob

import re
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
# from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import graphviz_layout


# ----------------------------------------------------------------------------------------------------------------------

def extract_arguments():
    """
    Extracting the provided arguments and set default values for non-provided arguments.
    :return: parser object
    """
    parser = argparse.ArgumentParser(description="In-Network Computing Emulator")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="""The number of iterations over the dataset.""")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="""The amount of samples for each batch.""")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                        help="""Hyperparamater that declares the sensitivity of the training process""")
    parser.add_argument("-ma", "--master_addr", type=str, default="10.0.0.1",
                        help="""IP address of the parameter-server.""")
    parser.add_argument("-mp", "--master_port", type=str, default="9999",
                        help="""Port number of the parameter-server""")
    parser.add_argument("-bw", "--bandwidth", type=float, default=10, help="""Bandwidth of in-network links, Default is
    10 MB/s.""")
    return parser.parse_args()


def read_topology(file_path):
    """
    :param file_path: Input file (.txt) of tree-based topology (adjacency list).
    :return: Dictionary of nodes and their neighbors in the topology.
    """
    adj_dct = {}
    with open(file_path, 'r') as topology_file:
        for line in topology_file:
            line = line.strip().split()
            node = int(line[0])
            neighbors = list(map(int, line[1:]))
            adj_dct[node] = neighbors
    return adj_dct


def read_deployment(file_path):
    """
    :param file_path: Input file (.txt) of smart switches (seperated by " ").
    :return: List of smart switches indices.
    """
    with open(file_path, 'r') as blue_sw_file:
        line = blue_sw_file.readline()
        smarts_indices = list(map(int, line.split()))
    return smarts_indices


def read_workload(file_path):
    """
    :param file_path: Input file (.txt) of the ML task workers (list of tuples converted to string)
    :return: Dictionary of switches that are connected to workers.
    """
    with open(file_path, 'r') as load_file:
        load_data = load_file.read()
    data = load_data.strip('[]').split('),(')
    load_lst, load_dct = [tuple(map(int, pair.strip('()').split(','))) for pair in data], {}
    for key, val in load_lst:
        load_dct[key] = val
    return load_dct


def get_subtree(node, adj_dct, smarts_indices):
    """
    DFS-based lookup, used to map for each worker his closest aggregator (smart-switch/ps)
    :param node: Current node in the lookup
    :param adj_dct: Dictionary of nodes and their neighbors in the topology.
    :param smarts_indices: List of smart nodes indices.
    :return: List of workers in the aggregator's subtree.
    """
    if node not in adj_dct:
        return [node]
    else:
        children = adj_dct[node]
        worker_list = []

        for child in children:
            if child not in smarts_indices:
                worker_list.extend(get_subtree(child, adj_dct, smarts_indices))
            else:
                worker_list.append(child)
        return worker_list


def update_networkx_tree(paths_dct, leaf_nodes):
    edges = []
    for path in paths_dct.values():
        for idx in range(len(path) - 1):
            if (path[idx + 1], path[idx]) not in edges:
                edges.append((path[idx + 1], path[idx]))

    edges = sorted(edges, key=lambda x: str(x[0]))
    G_new = nx.Graph()
    G_new.add_node('h0')
    # Add edges from the edge list
    for edge in edges:
        if edge[0] in leaf_nodes:
            G_new.add_node(edge[0], host=True)
        else:
            G_new.add_node(edge[0], host=False)
        if edge[1] in leaf_nodes:
            G_new.add_node(edge[1], host=True)
        else:
            G_new.add_node(edge[1], host=False)
        G_new.add_edge(edge[0], edge[1])
    G_new.nodes['h0']['host'] = True
    return G_new


def path_balancing(G, workers_count, smarts_count):
    leaf_nodes = [node for node in G.nodes() if G.degree(node) == 1 and node[1] != '0']
    paths = {}
    for node in leaf_nodes:
        extended_path = [node]
        sp = nx.shortest_path(G, source=node, target='h0')[:-1]
        for i, switch in enumerate(sp[1:]):
            if switch not in smarts_sw_indices:
                extended_path += [str(sp[i]) + 'A', str(sp[i]) + 'B']
            extended_path.append(switch)
        paths[node] = extended_path + ['h0']
    G2 = update_networkx_tree(paths, leaf_nodes)
    color = ["orange" if G2.nodes[node].get('host') else "blue" if node in smarts_sw_indices else "red" for node in
             G2.nodes]
    nx.draw(G2, graphviz_layout(G2, prog="dot"), with_labels=True, node_color=color)
    plt.savefig(f'Topology_Balanced_{workers_count}_Workers_{smarts_count}_Smarts.png')
    plt.close()
    return paths


def networkx_tree(adj_dct, workers_map):
    """
    Used to pre-process and maintain the data for the mininet emulation
    :return: NetworkX object of the network.
    """
    G = nx.Graph()
    G.add_node('h0', host=True)
    G.add_edge('h0', 0)
    for sw_idx in [0] + [sw_idx for val in adj_dct.values() for sw_idx in val]:
        G.add_node(sw_idx)

    for sw_idx in adj_dct:
        for neighbor in adj_dct[sw_idx]:
            G.add_edge(sw_idx, neighbor)

    curr_rank = 1
    for sw_idx in workers_map:
        for _ in range(workers_map[sw_idx]):
            G.add_node(f'h{curr_rank}', host=True)
            G.add_edge(f'h{curr_rank}', sw_idx)
            curr_rank += 1

    return G


def plot_graph(name, title, x, y, x_lbl, y_lbl, x_ticks_flag):
    """
    Plot results of accuracy and loss measured for each epoch.
    """
    plt.plot(x, y)
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    if x_ticks_flag: plt.xticks(range(1, len(x) + 1))
    plt.title(title)
    plt.savefig(name)
    plt.close()
    return


def process_csv(G):
    """
    Process the recorded time measurements of the hosts participated in the ML task.
    :param G: NetworkX object of the network
    """
    training_data_path = glob(f'training_data_*.txt')[0]
    with open(training_data_path) as train_file:
        epochs = int(train_file.readlines()[3].split(": ")[1])

    workers_df = pd.DataFrame()
    for rank_of_worker in range(1, num_of_workers + 1):
        worker_df = pd.DataFrame()
        for epoch in range(1, epochs + 1):
            ps_csv = glob(f'network_csv/epoch_{epoch}_root_{0}_rank_{0}.csv')[0]
            ps_df = pd.read_csv(ps_csv)
            csv = glob(f'network_csv/epoch_{epoch}_root_*_rank_{rank_of_worker}.csv')[0]
            worker_epoch_df = pd.read_csv(csv)
            worker_idx = f'h{rank_of_worker}'
            sp = nx.shortest_path(G, source=worker_idx, target='h0')
            closest_sw_idx = -1
            for vertex in sp[:-1][::-1]:
                if vertex in smarts_sw_indices:
                    closest_sw_idx = smarts_sw_indices.index(vertex)
                    break

            accumulated_agg_time = ps_df['aggregation_time']
            if closest_sw_idx != -1:
                main_grp_worker_rank = smart_rank_map[smarts_sw_indices[closest_sw_idx]]
                for node in sp[1:-1]:
                    if node in smarts_sw_indices:
                        agg_rank = smart_rank_map[node]
                        agg_csv = glob(f'network_csv/epoch_{epoch}_root_{agg_rank}_rank_{agg_rank}.csv')[0]
                        accumulated_agg_time += pd.read_csv(agg_csv)['aggregation_time']
            else:
                main_grp_worker_rank = worker_rank
            worker_epoch_df['agg_time'] = accumulated_agg_time
            worker_epoch_df['comm_time'] = ps_df[str(main_grp_worker_rank)] - worker_epoch_df['comm_batch_start'] - \
                                           worker_epoch_df['agg_time']

            worker_batch_series = worker_epoch_df.sum()
            new_df = pd.DataFrame(data={'rank': worker_rank, 'epoch_time': worker_batch_series['batch_time'],
                                        'comp_time': worker_batch_series['comp_time'],
                                        'comm_time': worker_batch_series['comm_time'],
                                        'agg_time': worker_batch_series['agg_time']}, index=[0])

            worker_df = pd.concat([worker_df, new_df])
        worker_epoch_series = worker_df.sum() / epochs
        total_worker_df = pd.DataFrame(
            data={'rank': worker_rank, 'avg_epoch_time': worker_epoch_series['epoch_time'],
                  'avg_comp_time': worker_epoch_series['comp_time'],
                  'avg_comm_time': worker_epoch_series['comm_time'],
                  'avg_agg_time': worker_epoch_series['agg_time']}, index=[0])
        workers_df = pd.concat([workers_df, total_worker_df])
    print(f'\n{workers_df}')
    run_avg_series = workers_df.drop(columns=['rank']).sum() / num_of_workers
    df = pd.DataFrame(data={'computation': [run_avg_series['avg_comp_time']],
                            'communication': [run_avg_series['avg_comm_time']],
                            'epoch_time': [run_avg_series['avg_epoch_time']],
                            'aggregation': [run_avg_series['avg_agg_time']]})
    df.to_csv(f'network_csv/barplot_data_{num_of_workers}_workers_{num_of_smarts}_smarts.csv')
    return


def pre_processing(adj_dct, load_dct, workers_count, smarts_count):
    G = networkx_tree(adj_dct, load_dct)
    node_color = ["orange" if G.nodes[node].get('host') else "blue" if node in smarts_sw_indices else "red" for node
                  in G.nodes]
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_color=node_color)
    plt.savefig(f'Topology_{workers_count}_Workers_{smarts_count}_Smarts.png')
    plt.close()
    return G


def post_processing(G):
    """
    Post-process the collected data during the emulation and export the results.
    """
    # epochs_lst = list(range(1, epochs_num + 1))
    # accumulated_time = [sum(time[:i]) for i in range(1, len(time) + 1)]
    # plot_graph("acc_per_epoch.png", "Accuracy Per Epoch", epochs_lst, acc, "Epoch", "Accuracy", 1)
    # plot_graph("acc_per_time.png", "Accuracy Per Time", accumulated_time, acc, "Time [sec]", "Accuracy", 0)
    # plot_graph("loss_per_epoch.png", "Loss Per Epoch", epochs_lst, loss, "Epoch", "Accuracy", 1)
    # plot_graph("loss_per_time.png", "Loss Per Time", accumulated_time, loss, "Time [sec]", "Accuracy", 0)
    # plot_graph("epoch_per_time.png", "Epoch Per Time", epochs_lst, accumulated_time, "Epoch", "Time [sec]", 1)
    process_csv(G)
    return


# ----------------------------------------------------------------------------------------------------------------------

def build_network(topology, hosts_sw_dct, workers_num, paths):
    """
    :param topology: Dictionary of nodes and their neighbors in the topology.
    :param hosts_sw_dct: Dictionary of switches that are connected to workers.
    :param workers_num: Total number of workers participating in the ML task.
    :param paths: List of paths to root from each worker
    """
    options = extract_arguments()
    print(options)
    print(f'Topology: {topology}\nHosts_Sw_Dct: {hosts_sw_dct}\nNum_Of_Workers: {workers_num}')
    net = Mininet(topo=None, build=False, link=TCLink, ipBase='10.0.0.0/8')

    info('\n*** Adding controller\n')
    con1 = net.addController(name='con1', controller=Controller, protocol='tcp', port=6633)

    switches, hosts, links, smart_links = [], [], [], []

    for path in paths:
        for obj in paths[path]:
            if type(obj) == str and obj[0] == 'h' and len(obj) == 2 and obj not in hosts:
                hosts.append(obj)
            elif obj not in switches:
                switches.append(obj)
                if obj in smarts_sw_indices:
                    hosts.append(f'h{smart_rank_map[obj]}')
                    smart_links.append((obj, f'h{smart_rank_map[obj]}'))
    for path_src in paths:
        for i in range(len(paths[path_src]) - 1):
            if (paths[path_src][i], paths[path_src][i + 1]) not in links:
                links.append((paths[path_src][i], paths[path_src][i + 1]))

    link_bw = options.bandwidth * 8
    mininet_hosts, mininet_switches = [], []

    info("\n*** Add Switches: ***\n")
    for switch in switches:
        mininet_switches.append(net.addSwitch(f'{switch}'))
        info(f'{switch} is added\n')

    info("\n*** Add Hosts: ***\n")
    for host in sorted(hosts, key=lambda h: int(re.search(r'\d+', h).group())):
        mininet_hosts.append(net.addHost(f'{host}'))
        info(f'host {host} is added\n')

    info("\n*** Add Links: ***\n")
    for link in links:
        net.addLink(f'{link[0]}', f'{link[1]}', bw=link_bw)
        info(f'{link[0]} <--> {link[1]} | bw=limited\n')
    for link in smart_links:
        net.addLink(f'{link[0]}', f'{link[1]}')
        info(f'{link[0]} <--> {link[1]} | bw=unlimited\n')

    info("\n--------------------\n")
    info('*** Starting network ***\n')
    net.build()
    info('\n*** Starting controllers ***\n')
    for controller in net.controllers:
        controller.start()

    for tree_sw in mininet_switches:
        tree_sw.start([con1])

    # ------------------------------------------------------------------------------------------------------------------
    info('\n*** Post configure switches and hosts\n\n')
    CRED, CEND, CGREEN = '\033[41m', '\033[0m', '\033[32m'

    info(f'\n{CGREEN}Topology{CEND}: {adj_map}\n')
    info(f'{CGREEN}Deployment{CEND}: {smarts_sw_indices}\n')
    info(f'{CGREEN}Load{CEND}: {load_map}\n')
    info(f'{CGREEN}Host Groups{CEND}: {rank_groups}\n\n')

    # Connectivity Check:
    net.pingAll()
    # net.iperfAll()

    # Background Note:
    info(
        f'\n\n{CRED}Note: The script is currently running in the background, the program will be closed automatically '
        f'when finished.{CEND}\n')

    # Initiate the distributed ML task:
    waiting_lst = []
    for i in range(world_size):
        p = net.get('h' + str(i)).popen(f"sudo python3 dist_ml.py {options.master_addr} {options.master_port} "
                                        f"{world_size} {i} {options.epochs} {options.batch_size} "
                                        f"{options.learning_rate} {str(group_mapping[i]).replace(' ', '')} "
                                        f"{num_of_smarts}")
        waiting_lst.append(p)

    # Waiting for all hosts to finish their job
    for p in waiting_lst:
        p.wait()

    # Uncomment to enable CLI:
    # CLI(net)

    # Stop the emulation:
    net.stop()
    return


# ----------------------------------------------------------------------------------------------------------------------
# Main Script:
# ----------------------------------------------------------------------------------------------------------------------

# Process Input Files:
# --------------------
topology_file_path = 'topology.txt'
deployment_file_path = 'deployment.txt'
workload_file_path = 'workload.txt'

adj_map = read_topology(topology_file_path)
smarts_sw_indices = read_deployment(deployment_file_path)
load_map = read_workload(workload_file_path)

workers_sw_indices = list(load_map.keys())
num_of_workers = sum(load_map[worker_sw] for worker_sw in workers_sw_indices)
num_of_smarts = len(smarts_sw_indices)
world_size = 1 + num_of_smarts + num_of_workers

hosts_sw_map = {0: [0]}
rank_counter = 0
for worker_sw_idx in workers_sw_indices:
    hosts_sw_map[worker_sw_idx] = []
    for worker in range(1, load_map[worker_sw_idx] + 1):
        rank_counter += 1
        hosts_sw_map[worker_sw_idx].append(rank_counter)

for smart_sw_idx in smarts_sw_indices:
    rank_counter += 1
    if smart_sw_idx == 0 or smart_sw_idx in workers_sw_indices:
        hosts_sw_map[smart_sw_idx].append(rank_counter)
    else:
        hosts_sw_map[smart_sw_idx] = [rank_counter]

groups = {}
if len(hosts_sw_map[0]) == 1:
    for smart_idx in [0] + smarts_sw_indices:
        if smart_idx not in workers_sw_indices:
            groups[smart_idx] = [smart_idx] + get_subtree(smart_idx, adj_map, smarts_sw_indices)
else:
    for smart_idx in smarts_sw_indices:
        if smart_idx not in workers_sw_indices:
            groups[smart_idx] = [smart_idx] + get_subtree(smart_idx, adj_map, smarts_sw_indices)

rank_groups = {}
for root_sw in groups.keys():
    rank_groups[hosts_sw_map[root_sw][0]] = []
    for worker_sw in groups[root_sw]:
        rank_lst = hosts_sw_map[worker_sw]
        if rank_lst[-1] not in [hosts_sw_map[i][-1] for i in smarts_sw_indices]:
            rank_groups[hosts_sw_map[root_sw][0]] += rank_lst
        else:
            rank_groups[hosts_sw_map[root_sw][0]] += [rank_lst[-1]]

for smart_sw_idx in smarts_sw_indices:
    if smart_sw_idx == 0:
        rank_groups[hosts_sw_map[smart_sw_idx][0]] = [0, hosts_sw_map[smart_sw_idx][1]]
        sub_group = [hosts_sw_map[smart_sw_idx][1]]

        for sw in groups[smart_sw_idx]:
            if sw == 0:
                continue
            elif sw in workers_sw_indices and sw in smarts_sw_indices:
                sub_group.append(hosts_sw_map[sw][-1])
            else:
                for worker_rank in hosts_sw_map[sw]:
                    sub_group.append(worker_rank)
        rank_groups[hosts_sw_map[smart_sw_idx][1]] = sub_group
    elif smart_sw_idx in workers_sw_indices:
        rank_groups[hosts_sw_map[smart_sw_idx][-1]] = [hosts_sw_map[smart_sw_idx][-1]] + hosts_sw_map[smart_sw_idx][:-1]

smart_rank_map = {}
for smart in smarts_sw_indices:
    smart_rank_map[smart] = world_size - num_of_smarts + smarts_sw_indices.index(smart)

group_mapping = {}
for root_rank in sorted(rank_groups.keys()):
    for rank in rank_groups[root_rank]:
        if rank in group_mapping:
            group_mapping[rank].append(rank_groups[root_rank])
        else:
            group_mapping[rank] = [rank_groups[root_rank]]

# Cleanup directory and initialize the emulation:
# -----------------------------------------------
os.system("rm -r error_logs")
os.system("rm -r output_logs")
os.system("rm -r network_csv")
os.system("rm -r tshark_logs")
os.system("rm -r wireshark_pcaps")
os.system("mkdir error_logs")
error_log = open("error_logs/err_net.txt", "w")
sys.stderr = error_log
os.system("mkdir wireshark_pcaps")
os.system("mkdir output_logs")
os.system("mkdir network_csv")
os.system("mkdir tshark_logs")
setLogLevel('info')
#
G_unbalanced = pre_processing(adj_map, load_map, num_of_workers, num_of_smarts)
paths_dict = path_balancing(G_unbalanced, num_of_workers, num_of_smarts)
build_network(adj_map, hosts_sw_map, num_of_workers, paths_dict)
post_processing(G_unbalanced)
# ----------------------------------------------------------------------------------------------------------------------
