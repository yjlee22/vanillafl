import os
import time
import numpy as np

import torch
from utils import *
from dataset import Dataset
from torch.utils import data
import torch.nn.functional as F
import gc

class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        # super(Server, self).__init__()
        self.args = args
        self.device = device
        self.datasets = datasets
        self.model_func = model_func
        
        self.server_model = init_model
        self.server_model_params_list = init_par_list 
        
        print("Initialize the Server      --->  {:s}".format(self.args.method))
        ### Generate Storage
        print("Initialize the Public Storage:")
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(
                self.clients_params_list.shape[0], self.clients_params_list.shape[1]))
        
        self.clients_updated_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        # self.clients_updated_params_list = np.expand_dims(init_par_list, axis=0).repeat(args.total_client, axis=0)
        print(" Local Updated Param List  --->  {:d} * {:d}".format(
                self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))

        ### Generate Log Storage : [[loss, acc]...] * T
        self.test_perf = np.zeros((self.args.comm_rounds, 2))
              
        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate
        
        # transfer vectors (must be defined if use)
        self.comm_vecs = {
            'Params_list': None,
        }
        self.received_vecs = None
        self.Client = None
        
    def _see_the_watch_(self):
        # see time
        self.time.append(datetime.datetime.now())
    
    def _activate_clients_(self, t):    
        return np.random.choice(range(self.args.total_client), max(int(self.args.active_ratio * self.args.total_client), 1), replace=True)
            
    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay
        
    def _test_(self, t, selected_clients):
        # validation on test set
        loss, acc = self._validate_((self.datasets.test_x, self.datasets.test_y))
        #import ipdb; ipdb.set_trace()
        self.test_perf[t] = [loss, acc]
        print("    Test    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(self.test_perf[t][0], self.test_perf[t][1]), flush = True)
        
    def _summary_(self):
        if not self.args.non_iid:
            summary_root = f'{self.args.out_file}/summary/{self.args.dataset}_IID'
        else:
            summary_root = f'{self.args.out_file}/summary/{self.args.dataset}_{self.args.split_rule}_{self.args.split_coef}'
            
        if not os.path.exists(summary_root):
            os.makedirs(summary_root)
        summary_file = summary_root + f'/{self.args.model}_{self.args.method}_lamb_{self.args.lamb}_alpha_{self.args.alpha}_rho_{self.args.rho}_beta_{self.args.beta}.txt'
        with open(summary_file, 'w') as f:
            f.write("##=============================================##\n")
            f.write("##                   Summary                   ##\n")
            f.write("##=============================================##\n")
            f.write("Communication round   --->   T = {:d}\n".format(self.args.comm_rounds))
            f.write("Average Time / round   --->   {:.2f}s \n".format(np.mean(self.time)))
            #erase
            f.write("Total HPU Hours   --->   {:.4f}h \n".format(np.sum(self.time[3:]) * 10 / 3600))
            f.write("Top-1 Test Acc (T)    --->   {:.2f}% ({:d})".format(np.max(self.test_perf[:,1]), np.argmax(self.test_perf[:,1])))    
        
        # print results summary
        print("##=============================================##")
        print("##                   Summary                   ##")
        print("##=============================================##")
        print("     Communication round   --->   T = {:d}       ".format(self.args.comm_rounds))
        print("    Average Time / round   --->   {:.2f}s        ".format(np.mean(self.time)))
        print("     Top-1 Test Acc (T)    --->   {:.2f}% ({:d}) ".format(np.max(self.test_perf[:,1]), np.argmax(self.test_perf[:,1])))
    
    
    def _validate_(self, dataset):
        self.server_model.eval()
        testdataset = data.DataLoader(Dataset(dataset[0], dataset[1], train=False, dataset_name=self.args.dataset, args=self.args), batch_size=32, shuffle=False)
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testdataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long()
                
                predictions = self.server_model(inputs)
                
                test_loss += F.cross_entropy(predictions, labels.squeeze(dim=-1), reduction='mean').item()

                y_pred = predictions.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

            test_acc = 100.00 * correct / len(testdataset.dataset)
                
        if self.args.weight_decay != 0.:
            # Add L2 loss
            test_loss += self.args.weight_decay / 2. * torch.sum(self.server_model_params_list * self.server_model_params_list)
        
        return test_loss / (i+1), test_acc
    
    
    def _save_results_(self):

        root = f'{self.args.out_file}'
        if not os.path.exists(root):
            os.makedirs(root)
        if not self.args.non_iid:
            root += f'/{self.args.dataset}_IID'
        else:
            root += f'/{self.args.dataset}_{self.args.split_rule}_{self.args.split_coef}'
        if not os.path.exists(root):
            os.makedirs(root)
        
        if not os.path.exists(root):
            os.makedirs(root)
        
        # save [loss, acc] results
        test_file = root + f'/{self.args.model}_{self.args.method}_lamb_{self.args.lamb}_alpha_{self.args.alpha}_rho_{self.args.rho}_beta_{self.args.beta}.npy'
        np.save(test_file, self.test_perf)
                        
    def process_for_communication(self):
        pass
        
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        pass
    
    def postprocess(self, client, received_vecs):
        pass
        
    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")
        
        Averaged_update = torch.zeros(self.server_model_params_list.shape)
        
        for t in range(self.args.comm_rounds):
            start = time.time()
            # select active clients list
            selected_clients = self._activate_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush = True)
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clients])))
            
            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)
                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs, dataset=dataset, lr=self.lr, args=self.args)
                self.received_vecs = _edge_device.train()
                self.clients_updated_params_list[client] = self.received_vecs['local_update_list']
                self.clients_params_list[client] = self.received_vecs['local_model_param_list']
                self.postprocess(client, self.received_vecs)
                
                # release the salloc
                del _edge_device
            
            # calculate averaged model
            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model  = torch.mean(self.clients_params_list[selected_clients], dim=0)
            
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_client_from_params(self.device, self.server_model, self.server_model_params_list)
            
            
            self._test_(t, selected_clients)
            self._lr_scheduler_()
            
            # time
            end = time.time()
            self.time[t] = end-start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush = True)
    
            
        
        self._save_results_()
        self._summary_()