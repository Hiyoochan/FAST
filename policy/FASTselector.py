import numpy as np
import os
import torch
from flemme.logger import get_logger
from flemme.logger import get_logger_to_file
from fast.dataio import create_auxtar_loader
from flemme.trainer_utils import freeze, unfreeze
import time

logger = get_logger('policy')
def create_policy(epoch, model, device, policy_config, auxiliary_dataset_names, rewards, selections_per_dataset, filter_per_dataset, MAB_config):
    policy_name = policy_config.pop('name', 'UCB')
    logger.info(f'following {policy_name}.')
    if policy_name == 'UCB':
        return UCBtrainer(epoch = epoch, model = model, device = device, 
            auxiliary_dataset_names = auxiliary_dataset_names,  rewards = rewards, selections_per_dataset = selections_per_dataset, filter_per_dataset = filter_per_dataset,
            MAB_config = MAB_config, **policy_config)
    

class UCBtrainer:
    def __init__(self, epoch, model, device, auxiliary_dataset_names, rewards, selections_per_dataset, filter_per_dataset, MAB_config):
        self.epoch = epoch
        self.model = model
        self.device = device
        self.reward_beta = MAB_config.get('reward_beta',0.9)
        self.alpha_T = MAB_config.get('alpha_T',0.1)
        self.loss_scaling = MAB_config.get('loss_scaling',1)
        self.loss_threshold = MAB_config.get('loss_threshold',10)
        self.filter_threshold = MAB_config.get('filter_threshold',20)
        self.auxiliary_dataset_names = auxiliary_dataset_names
        self.rewards = rewards
        self.selections_per_dataset = selections_per_dataset
        self.filter_per_dataset = filter_per_dataset
        self._initialize_all_args()
        log_file = MAB_config.get('aux_log_file')
        log_file_dir = os.path.dirname(log_file)
        os.makedirs(log_file_dir, exist_ok=True)
        self.save_logger = get_logger_to_file('selector', log_file)
        # self.save_logger = get_logger('selector')
        self.save_logger.info("Begin epoch {} MAB client selection:".format(epoch))


    def _initialize_all_args(self):
        # Initialize gradients, losses and rewards
        self._gradients = {name: None for name in self.auxiliary_dataset_names}
        self._sims = {name: None for name in self.auxiliary_dataset_names}
        self._losses = {name: None for name in self.auxiliary_dataset_names}
        # initialize UCB specific variables
        self._upper_confidence_index = {dataset_name: None for dataset_name in self.auxiliary_dataset_names}

    def update_reward(self, current_epoch, target_grad):
        # calculate similarity between grads and target
        for dataset_name in self.auxiliary_dataset_names:
            # if self._gradients[dataset_name] is not None:
            if self.filter_per_dataset[dataset_name] > self.filter_threshold:
                continue
            sim = self._calculate_similarity(target_grad, self._gradients[dataset_name])
            self._sims[dataset_name] = sim
            # self.save_logger.info("At current epoch {}, the cos_sim of {} to target is {}".format(current_epoch,dataset_name,sim))
            if current_epoch == 1:
                alpha = 0.5
            else:
                alpha = self._calculate_alpha(current_epoch) 
            reward = self._calculate_reward(sim, self._losses[dataset_name], alpha)
            # save_logger.info("At current epoch {}, the original reward of {} is {}".format(current_epoch,dataset_name,self.rewards[dataset_name]))
            self.rewards[dataset_name] = \
                (1 - self.reward_beta) * self.rewards[dataset_name] + \
                self.reward_beta * reward
            # self.save_logger.info("At current epoch {}, the updated reward of {} is {}".format(current_epoch,dataset_name,self.rewards[dataset_name]))
    def select_aux_dataset(self, epoch, model, aux_datasets, tar_datasets, last_selected_idx, loader_config):
        ###create target data loader
        # freeze(model.encoder)
        start_time = time.time()
        # end_time 
        tar_data_loader = create_auxtar_loader(tar_datasets, loader_config)
        # target_grad = self._calculate_grad(model, tar_data_loader).detach()
        target_grad, _ = self._calculate_grad_and_loss(model, tar_data_loader)
        for i, dataset_name in enumerate(self.auxiliary_dataset_names):
            # logger.info('creating auxility loader ...')
            aux_data_loader = create_auxtar_loader(aux_datasets[i], loader_config)
            # logger.info('calculate gradient and loss ...')
            self._gradients[dataset_name], self._losses[dataset_name] = \
                self._calculate_grad_and_loss(model, aux_data_loader)
            # self.save_logger.info("At current epoch {}, the loss of {} is {}".format(epoch,dataset_name,self._losses[dataset_name]))
        self.update_reward(epoch, target_grad)
        # UCB algorithm for multi-armed bandit problem
        played_rounds = epoch + len(self.auxiliary_dataset_names)
        dataset_loss_threshold = float('inf')
        best_action_idx, best_action_value = None, float('-inf')
        if last_selected_idx is not None:
            last_selected_dataset = self.auxiliary_dataset_names[last_selected_idx]
            dataset_loss_threshold = self._losses[last_selected_dataset] + self.loss_threshold
            self.save_logger.info("At current epoch {}, the loss threshold is {}".format(epoch,dataset_loss_threshold))
        for i, dataset_name in enumerate(self.auxiliary_dataset_names):
            if self.filter_per_dataset[dataset_name] > self.filter_threshold:
                continue
            if self.selections_per_dataset is None:
                val = self.rewards[dataset_name] + 0.1 * np.sqrt(2 * np.log(played_rounds))
            else:
                val = self.rewards[dataset_name] + 0.1 * np.sqrt(2 * np.log(played_rounds) / self.selections_per_dataset[dataset_name])
            self._upper_confidence_index[dataset_name] = val
            self.save_logger.info("{} Info: loss: {:.4f}, cos_sim: {:.4f}, reward: {:.4f}, UCB: {:.4f}.".format(dataset_name, self._losses[dataset_name], self._sims[dataset_name], self.rewards[dataset_name], self._upper_confidence_index[dataset_name]))
            # self.save_logger.info("At current epoch {}, the UCB of {} is {}".format(epoch,dataset_name,self._upper_confidence_index[dataset_name]))
            if epoch > 50 and self._losses[dataset_name] > dataset_loss_threshold:
                self.filter_per_dataset[dataset_name] += 1
                if self.filter_per_dataset[dataset_name] == self.filter_threshold:
                    # logger.info("Delete {} with large loss {} for filter {} times.".format(dataset_name, self._losses[dataset_name],self.filter_per_dataset[dataset_name]))
                    self.save_logger.info("Delete {} with large loss {} for filter {} times.".format(dataset_name, self._losses[dataset_name],self.filter_per_dataset[dataset_name]))
                else:
                    self.save_logger.info("Filter {} with large loss {} for {} times.".format(dataset_name, self._losses[dataset_name],self.filter_per_dataset[dataset_name]))
                
                continue
            if val > best_action_value:
                best_action_value = val
                best_action_idx = i

        best_dataset_name = self.auxiliary_dataset_names[best_action_idx]
        self.selections_per_dataset[best_dataset_name] += 1
        end_time = time.time()
        # unfreeze(model.encoder)
        self.save_logger.info("Start from epoch {}, select {} for joint training, selected {} times.".format(epoch,best_dataset_name,self.selections_per_dataset[best_dataset_name]))
        self.save_logger.info("Info: loss: {:.4f}, cos_sim: {:.4f}, reward: {:.4f}, UCB: {:.4f}.".format(self._losses[best_dataset_name], self._sims[best_dataset_name], self.rewards[best_dataset_name], self._upper_confidence_index[best_dataset_name]))
        logger.info("select {} for joint training, selected {} times".format(best_dataset_name,self.selections_per_dataset[best_dataset_name]))
        logger.info(f"MAB execution time: {end_time - start_time:.4f} seconds")
        return best_action_idx

    def _calculate_grad_and_loss(self, model, data_loader):
        all_grads = []
        # with torch.no_grad():
        # batch_count = 0
        # num_batches = 3

        total_samples = 0
        total_loss = 0
        for data in data_loader:
            # if batch_count >= num_batches:
            #     break
            x, y, _ = data
            x, y = x.to(self.device).float(), y.to(self.device)
            ## only use for data parallel
            # with model.no_sync():
            model.zero_grad()
            loss, _ = model.compute_loss(x, y)
            if type(loss) == list:
                loss = sum(loss)
            loss.backward()

            grads = []
            # accumulate the gradients
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach())
            grads = torch.cat([g.flatten() for g in grads]) 
            all_grads.append(grads)
            ### accumulate the loss
            if(model.loss_reduction == 'mean'):
                loss = loss * x.shape[0]
            total_loss += loss.item()
            total_samples += x.shape[0]

            # batch_count += 1

        mean_grads = torch.mean(torch.stack(all_grads), dim=0)
        mean_loss = total_loss / total_samples
        model.zero_grad()
        return mean_grads, mean_loss
    def _calculate_similarity(self, target_grad, auxiliary_grad):
        return torch.nn.functional.cosine_similarity(target_grad, auxiliary_grad, dim=0)

    def _calculate_alpha(self, current_epoch):
        return np.exp(-self.alpha_T * (current_epoch - 1))

    def _calculate_reward(self, sim, auxiliary_loss, alpha):
        return (1 - alpha) * sim - alpha * auxiliary_loss * self.loss_scaling