# In-Network Aggregation For Machine Learning
This repository contains the source code for a Python script that simulates a dynamic tree-based network topology. The topology is capable of performing In-Network Aggregation (INA) at specific locations. 

Once the network is created, the script executes a distributed machine learning (ML) task on the hosts and records the total training time. The distributed training follows a Parameter-Server (PS) architecture.

## How is it Done?
The network is emulated using the Mininet emulator, while the distributed machine learning task is implemented using the PyTorch framework.

![blockdiagram](https://github.com/bysaar/In-Network-Aggregation/assets/90688449/5f79771a-60f0-44be-b18e-c6daebc39382)

- Red nodes represents simple switches (store-and-forward)
- Blue nodes represents smart switches (INA-capable switch)

## Getting Started

To get started with the code on this repo, you need to either *clone* or *download* this repo into your machine as shown below;

```bash
git clone https://github.com/bysaar/In-Network-Aggregation
```

## Dependencies

Before you begin playing with the source code, you might need to install dependencies just as shown below;

```bash
pip3 install -r requirements.txt
```

requirements.txt:
```
matplotlib==3.6.3
mininet==2.3.0
networkx==3.0
pandas==1.5.3
torch==1.13.1+cpu
torchsummary==1.5.1
torchvision==0.14.1+cpu
```


## Run Instructions

To run this code you need to have three required inputs:
```
1. workload.txt - The amount and locations of the workers
2. topology.txt - The network structure and how the workers are connected to each other
3. deployment.txt - The amount and locations of the smart switches
```

** The repo contains a basic example with the format of these inputs. 


```bash
$sudo python3 net.py --arg1=val1 ...
```

#### Optional Arguments:

master_addr, master_port, bandwidth (MB), epochs, batch_size, learning_rate

```bash
$-> cd In-Network-Aggregation
$ In-Network-Aggregation-> sudo python3 net.py --arg1=val1 ...
```

Note: Tshark pcaps can be recorded during the run-time, if you are interested to record the network traffic all you need to do is to edit the `dist_ml.py` file and un-comment the lines with capture.start/capture.stop

## Example To Explore

#### Test Case:
<p align="center">
  <img src="https://github.com/bysaar/In-Network-Aggregation/assets/90688449/0ad690d8-f9a8-44f5-ad8a-ecfd3b470c51" width="480" height="480">
</p>

- Dataset: MNIST (Digits)
- Model: LeNet-5
- Bandwidth: 16 Mbps (Equal links)
- Workload: 31 Workers - [2,5,2,1,5,8,2,6]
- Topology: 4-Level Binary Tree

#### Deployment Algorithms:
- SOAR - Maximize Utilization (SAS@CoNext'21)
- SMC - Minimize Link Congestion (SAS@Infocoom'22)
- SMP - Minimize Path Congestion (SAS@In-prep)


#### Results:

![res1](https://github.com/bysaar/In-Network-Aggregation/assets/90688449/c5b6f0eb-6063-4c93-9b3e-1261be991f6d)


<br />   
<br />   


![res2](https://github.com/bysaar/In-Network-Aggregation/assets/90688449/56bde0e7-80db-4ee6-955d-93e269af0968)




