import torch
from .client import Client
from utils import *
from optimizer import *
# import gc
import habana_frameworks.torch.core as htcore

class fedgamma(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):   
        super(fedgamma, self).__init__(device, model_func, received_vecs, dataset, lr, args)

        # rebuild
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.optimizer = ESAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)

    def train(self):
        # local training
        self.model.train()
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                self.optimizer.paras = [inputs, labels, self.loss, self.model]
                self.optimizer.step()
                
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs['Local_VR_correction'].to(self.device)
                loss_correct = torch.sum(param_list * delta_list)
                
                loss_correct.backward()
                htcore.mark_step()
                
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                self.base_optimizer.step()
                htcore.mark_step()
                
                # del inputs, labels, param_list, delta_list, loss_correct
                # gc.collect()
                
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list
        
        return self.comm_vecs