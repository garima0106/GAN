#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:12:03 2020

@author: garima
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np

import ray
import argparse
import time

parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num-workers", default=3, type=int,
                    help="The number of workers to use.")
parser.add_argument("--num-parameter-servers", default=1, type=int,
                    help="The number of parameter servers to use.")
parser.add_argument("--dim", default=1000, type=int,
                    help="The number of parameters.")
parser.add_argument("--redis-address", default=None, type=str,
                    help="The Redis address of the cluster.")

args = parser.parse_args()

#helper functions 
def get_data_loader():
    """download dataset"""
    
    mnist_transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])
    
    #we add FileLock here because multiple workers will want to download data, and this may cause overwrite 
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader= torch.utils.data.DataLoader(datasets.MNIST(
            "~/data",
            train=True,
            download=True,
            transform=mnist_transforms),
            batch_size=128,shuffle=True)
        test_loader= torch.utils.data.DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=128, shuffle=True)
    return train_loader, test_loader

def evaluate(model, test_loader):
    """Evaluate the accuracy of the model on a validation set"""
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            #to execute faster
            if batch_idx*len(data)>1024:
                break
            outputs=model(data)
            _, predicted = torch.max(outputs.data,1)
            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    return 100.*correct/total


## Set up - define the Neural network
class ConvNet(nn.Module):
        """Small ConvNet for MNIST"""
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1=nn.Conv2d(1,3, kernel_size=3)
            self.fc= nn.Linear(192,10)
        
        def forward(self,x):
            x = F.relu(F.max_pool2d(self.conv1(x),3))
            x = x.view(-1,192)
            x = self.fc(x)
            return F.log_softmax(x,dim=1)
        
        def get_weights(self):
            return{k:v.cpu() for k,v in self.state_dict().items()}
        
        def set_weights(self, weights):
            self.load_state_dict(weights)
            
        def get_gradients(self):
            grads = []
            for p in self.parameters():
                grad = None if p.grad is None else p.grad.data.cpu().numpy()
                grads.append(grad)
            return grads
        
        def set_gradients(self, gradients):
            for g, p in zip(gradients, self.parameters()):
                if g is not None:
                    p.grad= torch.from_numpy(g)
           

#defining Parameter Server
            
@ray.remote(num_gpus=1)
class ParameterServer(object):
    def __init__(self, dim):
        self.params = np.zeros(dim, dtype=np.float32)

    def update_and_get_new_weights(self, *gradients):
        for grad in gradients:
            self.params += grad
        return self.params

    def ip(self):
        return ray.services.get_node_ip_address()

        
        
 
    
@ray.remote(num_gpus=1)
class Worker(object):
    def __init__(self, num_ps, dim):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.net = ConvNet(dim)
        self.num_ps = num_ps

    @ray.method(num_return_vals=args.num_parameter_servers)
    def compute_gradient(self, *weights):
        all_weights = np.concatenate(weights)
        self.net.set_weights(all_weights)
        gradient = self.net.get_gradients()
        if self.num_ps == 1:
            return gradient
        else:
            return np.split(gradient, self.num_ps)

    def ip(self):
        return ray.services.get_node_ip_address()


    
if __name__ == "__main__":
   # ignore_reinit_error=True
    if args.redis_address is None:
        # Run everything locally.
        ray.init(num_gpus=args.num_parameter_servers +  2*args.num_workers)
    else:
        # Connect to a cluster.
        ray.init(redis_address=args.redis_address)

    split_weights = np.split(np.zeros(args.dim, dtype=np.float32),
                             args.num_parameter_servers)

    # Create the workers.
    workers = [Worker.remote(args.num_parameter_servers, args.dim)
               for _ in range(args.num_workers)]

    # Create the parameter servers.
    pss = [ParameterServer.remote(split_weights[i].size)
           for i in range(args.num_parameter_servers)]

    # As a sanity check, make sure all workers and parameter servers are on
    # different machines.
    if args.redis_address is not None:
        all_ips = ray.get([ps.ip.remote() for ps in pss] +
                          [w.ip.remote() for w in workers])
        assert len(all_ips) == len(set(all_ips))

    while True:
        t1 = time.time()

 # Compute and apply gradients.
        assert len(split_weights) == args.num_parameter_servers
        grad_id_lists = [[] for _ in range(len(pss))]
        for worker in workers:
            gradients = worker.compute_gradient.remote(*split_weights)
            if len(pss) == 1:
                gradients = [gradients]

            assert len(gradients) == len(pss)
            for i in range(len(gradients)):
                grad_id_lists[i].append(gradients[i])

        # TODO(rkn): This weight should not be removed. Does it affect
        # performance?
        all_grad_ids = [grad_id for grad_id_list in grad_id_lists
                        for grad_id in grad_id_list]
        ray.wait(all_grad_ids, num_returns=len(all_grad_ids))

        t2 = time.time()

        split_weights = []
        for i in range(len(pss)):
            assert len(grad_id_lists[i]) == args.num_workers
            new_weights_id = pss[i].update_and_get_new_weights.remote(
                *(grad_id_lists[i]))
            split_weights.append(new_weights_id)

        # TODO(rkn): This weight should not be removed. Does it affect
        # performance?
        ray.wait(split_weights, num_returns=len(split_weights))

        t3 = time.time()
        print("elapsed times: ", t3 - t1, t2 - t1, t3 - t2)
        
        
        
        
        



