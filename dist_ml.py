# In-Network Aggregation: Distributed ML Source Code
# ----------------------------------------------------------------------------------------------------------------------

# Libraries:
# ----------
import os, sys, subprocess
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet18
from torchsummary import summary
from time import time as timer
import functools
import pandas as pd
import threading
import time


# ----------------------------------------------------------------------------------------------------------------------
# Classes:
# ----------------------------------------------------------------------------------------------------------------------

class TsharkCapture:
    """ T-shark link monitoring class. """

    def __init__(self, node_iface, node_type, node_rank):
        self.iface = node_iface
        self.type = node_type
        self.rank = node_rank
        self.filename = f"{self.type}_{self.rank}"
        self.tshark_error_log = open(f'tshark_logs/tshark_log_{self.rank}.txt', "a")
        self.tshark_error_log.write(f'Type: {self.type} | Interface: {self.iface} | Rank: {self.rank}\n\n')
        self.tshark_process = None

    def start(self, epoch):
        """
        Start tshark capture
        :return: None
        """
        if self.tshark_process is not None:
            raise ValueError("tshark process is already running.")

        cmd = ["tshark", "-w", f'wireshark_pcaps/Smarts_{smarts_count}_Epoch_{epoch}_{self.filename}.pcapng', "-F",
               "pcapng", "-i", f'{self.iface}', "-f", "tcp"]

        self.tshark_process = subprocess.Popen(cmd, stderr=self.tshark_error_log)

    def stop(self, epoch):
        """
        Stop capture and save the pccap file
        :return: None
        """
        time.sleep(1)
        if self.tshark_process is None:
            raise ValueError("tshark process is not running.")

        self.tshark_process.terminate()
        self.tshark_process.wait()
        self.tshark_process = None
        os.chmod(f'wireshark_pcaps/Smarts_{smarts_count}_Epoch_{epoch}_{self.filename}.pcapng', 0o777)

    def __del__(self):
        self.tshark_error_log.close()


class Node:
    """ Distributed ML general node implementation. """

    def __init__(self, rank, world_size, groups, smarts_count):
        self.rank = rank
        self.world_size = world_size
        self.groups = groups
        self.iface = 'h' + str(rank) + '-eth0'

        if rank == 0:
            self.type = "ps"
            self.agg_batch_time = -1
            self.num_params = -1
            self.main_global_dst = self.groups[0][0]
            self.group_workers = [worker for worker in self.groups[0] if worker != self.rank]
            self.comm_batch_end_arr = [[] for _ in self.group_workers]
        elif rank <= world_size - smarts_count - 1:
            self.type = "worker"
            self.comm_curr_batch_start = -1
            self.main_global_dst = self.groups[0][0]
        else:
            self.type = "smart"
            self.agg_batch_time = -1
            self.sub_global_dst = self.rank
            self.main_global_dst = self.groups[0][0]
            self.group_workers = [worker for worker in self.groups[1] if worker != self.rank]

        # Log Files
        self.error_log = open(f'error_logs/err_{self.rank}.txt', 'w')
        self.output_log = open(f'output_logs/out_{self.rank}.txt', 'w')

        # Tshark Capture Object
        self.capture = None

    def init_environment(self):
        """
        This method used to configure the environment and initialize the main process group for each process.
        :return: None
        """
        # Record Logfiles:
        sys.stdout = self.output_log
        sys.stderr = self.error_log

        # GLOO backend environment setup
        os.environ['MASTER_ADDR'] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ['GLOO_SOCKET_IFNAME'] = self.iface
        os.environ['TF_SOCKET_IFNAME'] = self.iface
        os.environ["GLOO_SOCKET_TIMEOUT"] = "3600"

        self.output_log.write(
            f'rank={self.rank}, iface={self.iface}, addr={master_addr}, port={master_port}, world_size={self.world_size}\n'
            f'groups={self.groups}\n')

        # Equal initial state:
        torch.manual_seed(0)

        # Init Tshark Capture:
        self.capture = TsharkCapture(self.iface, self.type, self.rank)

        # Process Group Initialization:
        print(f'Initializing {self.type}: rank: {self.rank} | world size: {self.world_size} | ParameterServerIP: '
              f'{master_addr}:{master_port} | Interface: {self.iface} | Groups: {self.groups}')
        dist.init_process_group(backend='gloo', init_method='env://', world_size=self.world_size, rank=self.rank)

        # Main group is a unique process group for workers / parameter server.
        # The subgroup is set only for smart switches.
        print(f'Process group successfully created.\n')
        print("-----------------------------------------------------")
        if self.type == "smart":
            print(f'Smart Global Rank: {self.rank}')
            print(
                f'Main Group: {self.groups[0]}\nGlobal Dst: {self.main_global_dst}\n')
            print(
                f'\nSub Group: {self.groups[1]}\nGlobal Dst: {self.sub_global_dst}\nWorkers: {self.group_workers}\n')
        else:
            print(f'{self.type} Global Rank: {self.rank}')
            print(f'Main Group: {self.groups}\nGlobal Dst: {self.main_global_dst}\n')
            if self.type == "ps": print(f'Workers: {self.group_workers}\n')
        print("-----------------------------------------------------")
        return

    def prepare_dataset(self, train=True):
        """
        This method handling all the data preparation, distributed sampling for workers and full load for the ps.
        :param isTrain: Boolean flag used to indicate which type of dataset is needed either train or test.
        :return: Dataset object, Dataloader object
        """
        # Preparation of dataset and appropriate dataloader.
        dataset_type = {True: 'Train', False: 'Test'}
        dataset = datasets.MNIST('../data', train=train, download=False, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        if self.type != "worker":
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        else:
            sampler = DistributedSampler(dataset=dataset, rank=self.rank - 1,
                                         num_replicas=world_size - 1 - smarts_count)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=True)
        if dataset and self.type != "worker":
            print(f'{dataset_type[train]} Dataset is loaded:\nTotal Batches: {len(loader)}\n')
        elif dataset and self.type == "worker":
            print(f'{dataset_type[train]} Dataset is loaded:\nShape: {dataset.data.shape}\nClasses: {dataset.classes}\n'
                  f'Total Batches={len(loader)}\nBatches Per Worker: {len(dataset) // (world_size - 1 - smarts_count)}\n')
        else:
            print(f'{dataset_type[train]} Dataset loading has been failed\n')
        return dataset, loader

    def average_gradients(self, model):
        """
        This method performs the gradient aggregation and synchronizes the result among the workers.
        :param model: Neural network object, Used to iterate over each parameter and transfer the matching gradient
        :return: None
        """
        # Using reduce & broadcast methods to transfer the gradients between ps and workers.
        # Both operations (reduce & broadcast) are synchronous (blocking state).
        for param_idx, param in enumerate(model.parameters()):
            if self.type == "smart":
                param.grad.data = self.measured_reduce(tensor=param.grad.data, dst=self.sub_global_dst, idx=param_idx)
                self.measured_reduce(tensor=param.grad.data, dst=self.main_global_dst, idx=param_idx)
                self.measured_broadcast(tensor=param.grad.data, src=self.main_global_dst, idx=param_idx)
                self.measured_broadcast(tensor=param.grad.data, src=self.sub_global_dst, idx=param_idx)
            else:  # ParameterServer / Worker
                if self.type == "ps":
                    param.grad.data = self.measured_reduce(tensor=param.grad.data, dst=self.main_global_dst,
                                                           idx=param_idx)
                    param.grad.data /= (world_size - smarts_count - 1)
                else:
                    self.measured_reduce(tensor=param.grad.data, dst=self.main_global_dst, idx=param_idx)
                self.measured_broadcast(tensor=param.grad.data, src=self.main_global_dst, idx=param_idx)
        return model

    def measured_send(self, tensor, dst, idx):
        """ :return: None"""
        if self.type == "worker" and idx == 0:
            self.comm_curr_batch_start = timer()
        dist.send(tensor=tensor, dst=dst)
        return

    def measured_recv(self, tensor, worker, idx):
        """ :return: None"""
        if self.type == "ps" and idx == self.num_params - 1:
            self.comm_batch_end_arr[self.group_workers.index(worker)].append(timer())
        dist.recv(tensor=tensor, src=worker)
        return

    def measured_broadcast(self, tensor, src, idx):
        """ :return: None"""
        if self.rank == src:
            send_threads = []
            for i, worker in enumerate(self.group_workers):
                send_threads.append(
                    threading.Thread(target=self.measured_send, args=(tensor, worker, idx)))
                send_threads[-1].start()
            for thread in send_threads:
                thread.join()
        else:
            self.measured_recv(tensor, src, idx)

    def measured_reduce(self, tensor, dst, idx):
        """ :return: The reduced tensor for the root in subtree."""
        if self.rank == dst:
            # Gather Operation:
            tensor, receive_threads = torch.zeros_like(tensor), []
            gathered_tensors = [tensor.clone() for _ in range(len(self.group_workers))]
            for i, worker in enumerate(self.group_workers):
                receive_threads.append(
                    threading.Thread(target=self.measured_recv, args=(gathered_tensors[i - 1], worker, idx)))
                receive_threads[-1].start()
            for thread in receive_threads:
                thread.join()
            # Reduce Operation (SUM):
            agg_start = timer()
            reduced_gradient = functools.reduce(lambda x, y: x + y, gathered_tensors)
            agg_time = timer() - agg_start
            self.agg_batch_time += agg_time
            return reduced_gradient
        else:
            self.measured_send(tensor=tensor, dst=dst, idx=idx)
            return


# ----------------------------------------------------------------------------------------------------------------------
# Models:
# ----------------------------------------------------------------------------------------------------------------------

# Imported Models:
# ----------------

def mnist_resnet50(model=resnet50()):
    # Import & edit Resnet50 neural network architecture to fit MNIST dataset:
    # Input Layer: Replace 3-dimension input layer to a single dimension one (RGB to Grayscale)
    # Output Layer: 10 Classes - One for each digit (0-9)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10, bias=True)
    return model


def mnist_resnet18(model=resnet18()):
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10, bias=True)
    return model


# ----------------------------------------------------------------------------------------------------------------------

# LeNet-5 Implementation:
# -----------------------

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2, stride=2), nn.Conv2d(6, 16, 5),
                                       nn.Tanh(), nn.AvgPool2d(2, stride=2))

        self.fc_model = nn.Sequential(nn.Linear(256, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
# Learning Functions
# ----------------------------------------------------------------------------------------------------------------------
def train(node, model, train_loader, optimizer, epoch, train_losses, train_counter, log_interval=10):
    loss_fn = nn.CrossEntropyLoss()
    train_loader.sampler.set_epoch(epoch)
    model.train()
    comp_time, batch_time, comm_batch_start = [], [], []
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = timer()
        # Local Computation
        # ----------------------------------------
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        comp_split = timer()
        # ----------------------------------------
        # Network: Communication + Synchronization
        # ----------------------------------------
        node.average_gradients(model=model)
        # ----------------------------------------
        optimizer.step()
        # ----------------------------------------
        batch_end = timer()
        comp_time.append(comp_split - batch_start)
        batch_time.append(batch_end - batch_start)
        comm_batch_start.append(node.comm_curr_batch_start)

        if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            # epoch, batch_idx * batch_size, len(train_loader.dataset) // (world_size - 1 - smart),
            # 100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 32) + ((epoch - 1) * (len(train_loader.dataset)) //
                                                     (world_size - 1 - smarts_count)))
    df_format = {'batch_time': batch_time, 'comp_time': comp_time, 'comm_batch_start': comm_batch_start}
    pd.DataFrame(data=df_format).to_csv(f'network_csv/epoch_{epoch}_root_{node.main_global_dst}_rank_{node.rank}.csv')
    return


def test(model, test_loader, test_losses, test_acc):
    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()  # reduction='sum'
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100 * correct / len(test_loader.dataset))

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    # .format(test_loss, correct, len(test_loader.dataset), 100 * correct / len(test_loader.dataset)))
    return


# ----------------------------------------------------------------------------------------------------------------------


def run_parameter_server(ps):
    """
    This function defines the behavior of the parameter server.
    """
    ps.init_environment()
    train_dataset, train_loader = ps.prepare_dataset(train=True)
    test_dataset, test_loader = ps.prepare_dataset(train=False)

    time_tracker = []
    model = LeNet()

    ps.num_params = len(list(model.parameters()))
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Model Parameters Overview:
    input_size = tuple(next(iter(train_dataset))[0].shape)
    summary(model, input_size)

    # Perform single batch learning to initialize parameters gradients
    images, labels = next(iter(train_loader))
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.zero_grad()

    # Checking the loss before training:
    test_losses, test_acc, test_counter = [], [], [i * len(train_loader.dataset) for i in range(epochs + 1)]
    test(model, test_loader, [], [])
    print("Parameter-Server initialization done.\n")
    # ------------------------------------------------------------------------------------------------------------------
    # Distribute average gradients
    print("Waiting for workers to transfer their gradients.")
    dist.barrier()

    for epoch in range(1, epochs + 1):
        # ps.capture.start(epoch)
        epoch_start = timer()
        agg_time = []
        for batch in range(len(train_loader) // (world_size - 1 - smarts_count)):
            ps.agg_batch_time = 0
            optimizer.zero_grad()
            ps.average_gradients(model=model)
            optimizer.step()
            agg_time.append(ps.agg_batch_time)

        time_tracker.append(timer() - epoch_start)
        df_format = {'aggregation_time': agg_time}
        for worker in ps.group_workers:
            df_format[worker] = ps.comm_batch_end_arr[ps.group_workers.index(worker)]

        pd.DataFrame(data=df_format).to_csv(f'network_csv/epoch_{epoch}_root_{ps.main_global_dst}_rank_{ps.rank}.csv')
        # ps.capture.stop(epoch)
        ps.comm_batch_end_arr = [[] for _ in ps.group_workers]
        test(model, test_loader, test_losses, test_acc)
    dist.barrier()
    print(f'{ps.type} rank {ps.rank} finished training.\n')
    fp = open(f'training_data_{smarts_count}.txt', 'w')
    fp.write(f"Accuracy: {test_acc}\nLoss: {test_losses}\nTimes: {time_tracker}\nEpochs: {epochs}")
    fp.close()
    ps.error_log.close()
    ps.output_log.close()
    return


def run_worker(worker):
    """
    This function defines the behavior of the worker.
    """
    worker.init_environment()
    train_subset, train_loader = worker.prepare_dataset(train=True)
    test_subset, test_loader = worker.prepare_dataset(train=False)
    model = LeNet()

    optimizer = optim.Adam(model.parameters(), learning_rate)
    train_losses, train_counter = [], []
    test_losses, test_acc, test_counter = [], [], [i * len(train_loader.dataset) for i in range(epochs + 1)]
    print("Model is ready for training\n")
    print(f"{worker.type} rank={worker.rank}\n\n")
    dist.barrier()

    for epoch in range(1, epochs + 1):
        # worker.capture.start(epoch)
        train(worker, model, train_loader, optimizer, epoch, train_losses, train_counter)
        # worker.capture.stop(epoch)
    test(model, test_loader, test_losses, test_acc)
    dist.barrier()
    print(f'{worker.type} rank {worker.rank} finished training.\n')
    worker.error_log.close()
    worker.output_log.close()
    return


def run_smart_switch(smart_sw):
    """
    This function defines the behavior of the smart switch.
    """
    smart_sw.init_environment()
    train_dataset, train_loader = smart_sw.prepare_dataset(train=True)
    model = LeNet()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Model Parameters Overview:
    input_size = tuple(next(iter(train_dataset))[0].shape)
    summary(model, input_size)

    # Perform single batch learning to initialize parameters gradients
    images, labels = next(iter(train_loader))
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.zero_grad()
    # ------------------------------------------------------------------------------------------------------------------
    # Distribute average gradients
    dist.barrier()
    for epoch in range(1, epochs + 1):
        # smart_sw.capture.start(epoch)
        agg_time = []
        for batch in range(len(train_loader) // (world_size - 1 - smarts_count)):
            smart_sw.agg_batch_time = 0
            optimizer.zero_grad()
            smart_sw.average_gradients(model=model)
            optimizer.step()
            agg_time.append(smart_sw.agg_batch_time)

        # Record time as virtual ps (subgroup)
        df_sub_format = {'aggregation_time': agg_time}
        pd.DataFrame(data=df_sub_format).to_csv(f'network_csv/epoch_{epoch}_root_{smart_sw.sub_global_dst}_rank_'
                                                f'{smart_sw.rank}.csv')
        # smart_sw.capture.stop(epoch)

    # print(f'Distributed training done. | Total time elapsed: {end_time - start_time} [Seconds].\n')
    dist.barrier()
    print(f'{smart_sw.type} rank {smart_sw.rank} finished training.\n')
    smart_sw.error_log.close()
    smart_sw.output_log.close()
    return


# ----------------------------------------------------------------------------------------------------------------------
# Main Script:
# ----------------------------------------------------------------------------------------------------------------------
master_addr, master_port = sys.argv[1], sys.argv[2]
epochs, batch_size = int(sys.argv[5]), int(sys.argv[6])
learning_rate = float(sys.argv[7])
world_size, rank = int(sys.argv[3]), int(sys.argv[4])
groups = eval(sys.argv[8])
smarts_count = int(sys.argv[9])

node = Node(rank, world_size, groups, smarts_count)
if node.type == "ps":
    run_parameter_server(node)
elif node.type == "worker":
    run_worker(node)
else:
    run_smart_switch(node)
# ----------------------------------------------------------------------------------------------------------------------

