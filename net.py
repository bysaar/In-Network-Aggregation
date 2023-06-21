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
import sys, os, argparse
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from glob import glob


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
    :param node: Current switch in the lookup
    :param adj_dct: Dictionary of nodes and their neighbors in the topology.
    :param smarts_indices: List of smart switches indices.
    :return: List of workers in the aggregator's subtree.
    """
    if node not in adj_map:
        return [node]
    else:
        left, right = adj_dct[node]
        if left not in smarts_indices and right not in smarts_indices:
            return get_subtree(left, adj_dct, smarts_indices) + get_subtree(right, adj_dct, smarts_indices)
        elif left not in smarts_indices:
            return get_subtree(left, adj_dct, smarts_indices) + [right]
        elif right not in smarts_indices:
            return [left] + get_subtree(right, adj_dct, smarts_indices)
        else:
            return [left] + [right]


def complete_tree(worker_switches, smart_switches, workers_map):
    """
    Used to pre-process and maintain the data for the mininet emulation
    :return: NetworkX object of the network.
    """
    num_of_leaves = len(worker_switches)
    # Calculate the total number of nodes in the tree
    num_of_nodes = 2 * num_of_leaves - 1
    G = nx.Graph()
    # Add nodes to the graph
    G.add_node('h0', host=True)
    for i in range(num_of_nodes):
        G.add_node(i, host=False)
    # Add edges to create a complete tree
    for i in range(num_of_leaves):
        if i == 0: G.add_edge('h0',i)
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < num_of_nodes:
            G.add_edge(i, left_child)
        if right_child < num_of_nodes:
            G.add_edge(i, right_child)
        if left_child in worker_switches:
            hosts = workers_map[left_child] if left_child not in smart_switches else workers_map[left_child][:-1]
            for host in hosts:
                G.add_node('h'+str(host),host=True)
                G.add_edge('h'+str(host),left_child)
        if right_child in worker_switches:
            hosts = workers_map[right_child] if right_child not in smart_switches else workers_map[right_child][:-1]
            for host in hosts:
                G.add_node('h'+str(host),host=True)
                G.add_edge('h'+str(host),right_child)
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
    for worker_rank in range(1, num_of_workers + 1):
        worker_df = pd.DataFrame()
        for epoch in range(1, epochs + 1):
            ps_csv = glob(f'network_csv/epoch_{epoch}_root_{0}_rank_{0}.csv')[0]
            ps_df = pd.read_csv(ps_csv)
            csv = glob(f'network_csv/epoch_{epoch}_root_*_rank_{worker_rank}.csv')[0]
            worker_epoch_df = pd.read_csv(csv)
            worker_idx = f'h{worker_rank}'
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
            worker_epoch_df['comm_time'] = ps_df[str(main_grp_worker_rank)] - worker_epoch_df['comm_batch_start'] - worker_epoch_df['agg_time']
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


def post_processing():
    """
    Post-process the collected data during the emulation and export the results.
    """
    #epochs_lst = list(range(1, epochs_num + 1))
    #accumulated_time = [sum(time[:i]) for i in range(1, len(time) + 1)]
    #plot_graph("acc_per_epoch.png", "Accuracy Per Epoch", epochs_lst, acc, "Epoch", "Accuracy", 1)
    #plot_graph("acc_per_time.png", "Accuracy Per Time", accumulated_time, acc, "Time [sec]", "Accuracy", 0)
    #plot_graph("loss_per_epoch.png", "Loss Per Epoch", epochs_lst, loss, "Epoch", "Accuracy", 1)
    #plot_graph("loss_per_time.png", "Loss Per Time", accumulated_time, loss, "Time [sec]", "Accuracy", 0)
    #plot_graph("epoch_per_time.png", "Epoch Per Time", epochs_lst, accumulated_time, "Epoch", "Time [sec]", 1)

    G = complete_tree(workers_sw_indices, smarts_sw_indices, hosts_sw_map)
    node_color = ["orange" if G.nodes[node].get('host') else "blue" if node in smarts_sw_indices else "red" for node
                  in G.nodes]
    pos = graphviz_layout(G, prog="dot", root=0)
    nx.draw(G, pos, with_labels=True, node_color=node_color)
    plt.savefig(f'Topology_{num_of_workers}_Workers_{num_of_smarts}_Smarts.png')
    process_csv(G)
    return


# ----------------------------------------------------------------------------------------------------------------------

def build_network(topology, hosts_sw_dct, num_of_workers):
    """
    :param topology: Dictionary of nodes and their neighbors in the topology.
    :param hosts_sw_dct: Dictionary of switches that are connected to workers.
    :param num_of_workers: Total number of workers participating in the ML task.
    """
    switches, hosts, red_switches_replica = [], [], []
    options = extract_arguments()
    print(options)
    net = Mininet(topo=None, build=False, link=TCLink, ipBase='10.0.0.0/8')

    info('\n*** Adding controller\n')
    con1 = net.addController(name='con1', controller=Controller, protocol='tcp', port=6633)

    info('\n*** Add switches\n')
    for sw_idx in [0] + [element for pair in topology.values() for element in pair]:
        switches.append(net.addSwitch('sw' + str(sw_idx)))
        info(f'sw{sw_idx} is added\n')

    info('\n*** Adding hosts\n')
    for sw_idx in hosts_sw_dct:
        for host in hosts_sw_dct[sw_idx]:
            hosts.append(net.addHost('h' + str(host)))
            info(f'h{host} is added\n')
            if host > num_of_workers:
                net.addLink(switches[sw_idx], hosts[-1]) # ,bw=options.bandwidth * 8 * 2)
            else:
                net.addLink(switches[sw_idx], hosts[-1],bw=options.bandwidth * 8)
            info(f'h{host} <--> sw{sw_idx}\n')

    info('\n*** Add links\n')
    # SW <--> SW:
    for sw_idx in topology.keys():
        for neighbor in topology[sw_idx]:
            if sw_idx in smarts_sw_indices:
                if neighbor in smarts_sw_indices or (neighbor not in smarts_sw_indices and neighbor not in workers_sw_indices):
                    net.addLink(switches[sw_idx], switches[neighbor], bw=options.bandwidth * 8)
                    info(f'sw{sw_idx} <--> sw{neighbor}\n')

    info("\n--------------------\n")
    info('*** Starting network\n')
    net.build()
    info('\n*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info('*** Starting switches\n')
    replica_switches = []

    for i in range(len(switches)//2):
        # Path-Balancing:
        # ---------------
        if i not in smarts_sw_indices:
            replica_switches.append(net.addSwitch('sw'+ str(i) + 'la'))
            replica_switches.append(net.addSwitch('sw' + str(i) + 'lb'))
            net.addLink(switches[i], replica_switches[-2], bw=options.bandwidth * 8)
            net.addLink(replica_switches[-2], replica_switches[-1], bw=options.bandwidth * 8)
            if 2*i+1 in workers_sw_indices and 2*i+1 not in smarts_sw_indices:
                replica_switches.append(net.addSwitch('sw' + str(i) + 'lc'))
                replica_switches.append(net.addSwitch('sw' + str(i) + 'ld'))
                net.addLink(replica_switches[-3], replica_switches[-2], bw=options.bandwidth * 8)
                net.addLink(replica_switches[-2], replica_switches[-1], bw=options.bandwidth * 8)
            net.addLink(replica_switches[-1], switches[2 * i + 1], bw=options.bandwidth * 8)

            replica_switches.append(net.addSwitch('sw'+ str(i) + 'ra'))
            replica_switches.append(net.addSwitch('sw'+ str(i) + 'rb'))
            net.addLink(switches[i], replica_switches[-2], bw=options.bandwidth * 8)
            net.addLink(replica_switches[-2], replica_switches[-1], bw=options.bandwidth*8)
            if 2*i+2 in workers_sw_indices and 2*i+2 not in smarts_sw_indices:
                replica_switches.append(net.addSwitch('sw' + str(i) + 'rc'))
                replica_switches.append(net.addSwitch('sw' + str(i) + 'rd'))
                net.addLink(replica_switches[-3], replica_switches[-2], bw=options.bandwidth * 8)
                net.addLink(replica_switches[-2], replica_switches[-1], bw=options.bandwidth * 8)
            net.addLink(replica_switches[-1], switches[2 * i + 2], bw=options.bandwidth * 8)
        else:
            if 2*i+1 not in smarts_sw_indices and 2*i+1 in workers_sw_indices:
                replica_switches.append(net.addSwitch('sw' + str(i) + 'la'))
                replica_switches.append(net.addSwitch('sw' + str(i) + 'lb'))
                net.addLink(switches[i], replica_switches[-2], bw=options.bandwidth * 8)
                net.addLink(replica_switches[-2], replica_switches[-1], bw=options.bandwidth * 8)
                net.addLink(replica_switches[-1], switches[2 * i + 1], bw=options.bandwidth * 8)
            if 2*i+2 not in smarts_sw_indices and 2*i+2 in workers_sw_indices:
                info(f"-->{2*i+2}\n")
                replica_switches.append(net.addSwitch('sw' + str(i) + 'ra'))
                replica_switches.append(net.addSwitch('sw' + str(i) + 'rb'))
                net.addLink(switches[i], replica_switches[-2], bw=options.bandwidth * 8)
                net.addLink(replica_switches[-2], replica_switches[-1], bw=options.bandwidth * 8)
                net.addLink(replica_switches[-1], switches[2 * i + 2], bw=options.bandwidth * 8)

    for tree_sw in switches:
        tree_sw.start([con1])
    for replica_sw in replica_switches:
        replica_sw.start([con1])

    # ------------------------------------------------------------------------------------------------------------------
    info('\n*** Post configure switches and hosts\n\n')
    CRED, CEND, CGREEN = '\033[41m', '\033[0m', '\033[32m'

    info(f'\n{CGREEN}Topology{CEND}: {adj_map}\n')
    info(f'{CGREEN}Deployment{CEND}: {smarts_sw_indices}\n')
    info(f'{CGREEN}Load{CEND}: {load_map}\n')
    info(f'{CGREEN}Host Groups{CEND}: {rank_groups}\n\n')

    # Debug:
    #info(f'{CGREEN}Group Mapping{CEND}: {group_mapping}\n')

    # Connectivity Check:
    net.pingAll()
    #net.iperfAll()

    # Background Note:
    info(
        f'\n\n{CRED}Note: The script is currently running in the background, the program will be closed automatically '
        f'when finished.{CEND}\n')

    # Initiate the distributed ML task:
    waiting_lst = []
    for i in range(world_size):
        p = net.get('h' + str(i)).popen(
            f"sudo python3 dist_ml.py {options.master_addr} {options.master_port} {world_size} "
            f"{i} {options.epochs} {options.batch_size} {options.learning_rate} "
            f"{str(group_mapping[i]).replace(' ', '')} {num_of_smarts}")
        waiting_lst.append(p)

    # Waiting for all hosts to finish their job
    for p in waiting_lst:
        p.wait()

    # Disable the Mininet CLI:
    #CLI(net)

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
            if sw == 0: continue
            elif sw in workers_sw_indices and sw in smarts_sw_indices:
                sub_group.append(hosts_sw_map[sw][-1])
            else:
                for worker_rank in hosts_sw_map[sw]:
                    sub_group.append(worker_rank)
        rank_groups[hosts_sw_map[smart_sw_idx][1]] = sub_group
    elif smart_sw_idx in workers_sw_indices:
        rank_groups[hosts_sw_map[smart_sw_idx][-1]] = [hosts_sw_map[smart_sw_idx][-1]]+hosts_sw_map[smart_sw_idx][:-1]

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
error_log = open("error_logs/err_net.txt","w")
sys.stderr = error_log
os.system("mkdir wireshark_pcaps")
os.system("mkdir output_logs")
os.system("mkdir network_csv")
os.system("mkdir tshark_logs")
setLogLevel('info')
build_network(adj_map, hosts_sw_map, num_of_workers)
post_processing()
# ----------------------------------------------------------------------------------------------------------------------
