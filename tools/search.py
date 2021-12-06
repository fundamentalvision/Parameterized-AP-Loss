import argparse
import datetime
import json
import random
import os
import math
import time
import matplotlib.pyplot as plt
import contextlib
import logging
import numpy as np
import torch
import search_util.misc as utils
import torch.multiprocessing as mp
ctx = mp.get_context("spawn")

from pathlib import Path
from torch.utils.data import DataLoader

from mmcv.utils import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmcv.runner import build_optimizer
from mmcv.parallel import MMDistributedDataParallel
from mmdet.utils import get_root_logger

# suppress warning
import warnings
warnings.filterwarnings("ignore")
# suppress log from mmdetection
logger = get_root_logger()
logger.setLevel(logging.ERROR)


def get_args_parser():
    parser = argparse.ArgumentParser('Parmaterized AP Loss search', add_help=False)
    parser.add_argument('--cfg_path', default='', 
                        help='path to config file')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='', 
                        help='resume from dir')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local_rank only needed for torch.distributed.launch')
    return parser


class TruncatedNormal(object):
    def __init__(self, mu, sigma, a, b):
        super(TruncatedNormal, self).__init__()
        # mu, sigma : (n_param,)
        # a, b : float
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

        self.n_params = len(mu)

        assert self.mu.dim() == 1
        assert b > a

        self.normal_sampler = torch.distributions.normal.Normal(
            torch.zeros(self.n_params, device=torch.cuda.current_device()),
            torch.ones(self.n_params, device=torch.cuda.current_device()))
        self.uniform_sampler = torch.distributions.uniform.Uniform(
            torch.zeros(self.n_params, device=torch.cuda.current_device()),
            torch.ones(self.n_params, device=torch.cuda.current_device()))

    def sample(self, batch_size):
        alpha = (-self.mu + self.a) / self.sigma
        beta = (-self.mu + self.b) / self.sigma

        uniform = self.uniform_sampler.sample([batch_size])

        alpha_normal_cdf = self.normal_sampler.cdf(alpha)
        p = (self.normal_sampler.cdf(beta) - alpha_normal_cdf).view(1, -1) * uniform + alpha_normal_cdf.view(1, -1)

        epsilon = torch.finfo(p.dtype).eps
        v = torch.clamp(2 * p - 1, -1 + epsilon, 1 - epsilon)

        x = self.mu.view(1, -1) + (2 ** 0.5) * torch.erfinv(v) * self.sigma.view(1, -1)
        x = torch.clamp(x, self.a, self.b)

        return x.detach()

    def log_prob(self, x):
        alpha = (-self.mu + self.a) / self.sigma
        beta = (-self.mu + self.b) / self.sigma

        normal_x = (x - self.mu) / self.sigma

        down = (self.normal_sampler.cdf(beta) - self.normal_sampler.cdf(alpha)).log().view(1, -1)

        up = (1.0 / self.sigma).log() + self.normal_sampler.log_prob(normal_x)

        l_prob = up - down

        l_prob = l_prob.where((x >= self.a) & (x <= self.b), torch.tensor(-np.inf, device=torch.cuda.current_device()))

        return l_prob
    

class Parametrized_AP_Loss:
    def __init__(self, args, cfg):
        self.args = args
        self.search_cfg = cfg 
        self.sampler = None 
        
        # build dataset
        dataset_train = build_dataset(self.search_cfg.data.train)
        dataset_val = build_dataset(self.search_cfg.data.val, dict(test_mode=True))
        self.data_loader_train = build_dataloader(dataset_train,
                                                  self.search_cfg.data.samples_per_gpu,
                                                  self.search_cfg.data.workers_per_gpu,
                                                  dist=args.distributed,
                                                  seed=args.seed)
        self.data_loader_val = build_dataloader(dataset_val,
                                                  1,
                                                  self.search_cfg.data.workers_per_gpu,
                                                  dist=args.distributed,
                                                  shuffle=False)
        # search process
        self.num_samples = cfg.num_samples
        self.num_theta = self.search_cfg.num_theta
        
        assert len(cfg.mu) == self.num_theta
        assert len(cfg.sigma) == self.num_theta
        
        mu_x = torch.tensor(cfg.mu, dtype=torch.float, device=torch.cuda.current_device())
        mu_x = (-torch.log(1.0/mu_x - 1))
        self.mu_x = mu_x.requires_grad_()
        self.sigma = torch.tensor(cfg.sigma, dtype=torch.float, device=torch.cuda.current_device())
    
    def search(self):
        print('Start searching, work dir: ', self.args.output_dir)
        rounds = []
        rewards = []
        
        # save mu as .npy each round for resume
        save_mu = self.mu_x.clone().view(-1).detach()
        save_mu = save_mu.sigmoid()
        save_mu = save_mu.cpu().numpy()
        save_mu_list = [save_mu]    
        save_sigma_list = [self.sigma.clone().detach().cpu().numpy()]
        save_theta_list = []
        save_reward_list = []
        
        already_run = 0
        if args.resume:
            save_mu_list = np.load(os.path.join(args.resume, 'mu.npy'))
            save_sigma_list = np.load(os.path.join(args.resume, 'sigma.npy'))
            save_theta_list = np.load(os.path.join(args.resume, 'sample_theta.npy'))
            save_reward_list = np.load(os.path.join(args.resume, 'sample_reward.npy'))
            already_run = save_mu_list.shape[0] - 1
            
            # a hack way to re run the last saved round
            already_run = already_run
            print('resume from round %d' % already_run)
            mu_x = torch.tensor(save_mu_list[-1], dtype=torch.float, device=torch.cuda.current_device())
            mu_x = (-torch.log(1.0/mu_x - 1))
            self.mu_x = mu_x.requires_grad_()
            self.sigma = torch.tensor(save_sigma_list[-1], device=torch.cuda.current_device())
            
            save_mu_list = save_mu_list.tolist()
            save_sigma_list = save_sigma_list.tolist()
            save_theta_list = save_theta_list.tolist()
            save_reward_list = save_reward_list.tolist()
        
        for s_t in range(already_run, self.search_cfg.sample_times):
            print('------Sample Round %02d------' % s_t)
            theta_groups, reward_groups = self.sample()
            
            mean_reward = reward_groups.mean()
            print(f"sample thetas: {theta_groups}")
            print(f"sample rewards: {reward_groups}")
            print(f"mean reward: {mean_reward}")
            
            save_reward_groups = reward_groups
            save_reward_groups = save_reward_groups.view(-1, 1).detach().cpu().numpy()
            save_reward_list.append(save_reward_groups)
            if torch.distributed.get_rank() == 0:
                save_reward_array = np.array(save_reward_list)
                save_reward_path = os.path.join(self.args.output_dir, 'sample_reward.npy')
                np.save(save_reward_path, save_reward_array)
            
            reward_groups = reward_groups - mean_reward
            reward_groups = reward_groups.view(-1, 1).detach()
            
            last_sigma = self.sigma
            # update sigma before PPO update
            self.sigma = self.sigma * (self.search_cfg.sample_times-s_t-1)/(self.search_cfg.sample_times-s_t)
        
            if self.sigma.sum() == 0: # hack for last round update
                self.sigma = last_sigma
            
            # PPO update
            argmax(self.mu_x, last_sigma, self.sigma, theta_groups, reward_groups, self.search_cfg, s_t)
            
             # visualization
            rounds.append(s_t)
            rewards.append(mean_reward)
            
            save_mu = self.mu_x.clone().view(-1).detach()
            save_mu = save_mu.sigmoid()
            save_mu = save_mu.cpu().numpy()
            save_mu_list.append(save_mu)

            save_sigma_list.append(self.sigma.clone().detach().cpu().numpy())
            save_theta_list.append(theta_groups.clone().detach().cpu().numpy())

            save_result(save_mu_list, save_sigma_list, save_theta_list, rounds, rewards, self.args.output_dir)

            print_mu = self.mu_x.clone().detach()
            print_mu = print_mu.sigmoid()
            print('Updated mu: ', print_mu)
            print('Updated sigma: ', self.sigma)
            print('\n')
            torch.distributed.barrier()    
        print('Finish the search process.')
        
    def sample(self):
        print('Sampling ......')
        print_sample_mu = self.mu_x.clone().detach()
        print_sample_mu = print_sample_mu.sigmoid()
        print('mu: ', print_sample_mu)
        print('sigma: ', self.sigma)
        
        self.sampler = TruncatedNormal(self.mu_x.sigmoid(), self.sigma, 0.0, 1.0)

        theta_groups = []
        reward_groups = []

        while len(theta_groups) < self.num_samples:
            theta, reward = self.sample_once()
            theta_groups.append(theta.detach())
            reward_groups.append(reward.detach())

        theta_groups = torch.cat(theta_groups, dim=0)
        reward_groups = torch.cat(reward_groups, dim=0)

        return theta_groups, reward_groups
    
    def sample_once(self):
        # build up model details
        device = torch.cuda.current_device()
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                model = build_detector(self.search_cfg.model, train_cfg=self.search_cfg.train_cfg, test_cfg=self.search_cfg.test_cfg)
        model.to(device)

        # build up optimizer
        optimizer = build_optimizer(model, self.search_cfg.optimizer)

        # build up scheduler
        lr_scheduler = utils.WarmUpScheduler(optimizer, self.search_cfg)
        
        if args.distributed:
            model = MMDistributedDataParallel(model,
                                              device_ids=[torch.cuda.current_device()],
                                              broadcast_buffers=False,
                                              find_unused_parameters=False)
        # sample parameters
        self.theta = torch.zeros([1, self.search_cfg.num_theta], device=torch.cuda.current_device())
        self.theta = self.sampler.sample(1)
        
        # the last element of theta decides the grad scale
        if self.search_cfg.model.type == 'RetinaNet':
            self.grad_scale = self.theta[:, -1] * 2 - 1
            self.grad_scale_real = torch.pow(10.0, self.grad_scale)
            self.theta = self.theta[:, :-1].view(1, self.search_cfg.search_func_num, -1) #[n_samples_per_gpu=1, n_func, n_para]
        elif self.search_cfg.model.type == 'FasterRCNN':
            self.grad_scale_rpn = self.theta[:, -2] * 2 - 1
            self.grad_scale_rcnn = self.theta[:, -1] * 2 - 1
            
            self.grad_scale_rpn_real = torch.pow(10.0, self.grad_scale_rpn)
            self.grad_scale_rcnn_real = torch.pow(10.0, self.grad_scale_rcnn)
            self.theta = self.theta[:, :-2].view(1, self.search_cfg.search_func_num, -1) #[n_samples_per_gpu=1, n_func, n_para]
        self.ctrl_points = self.convert_theta_coordinate(self.theta)
        
        torch.distributed.broadcast(self.ctrl_points, 0)
        if self.search_cfg.model.type == 'RetinaNet':
            torch.distributed.broadcast(self.grad_scale_real, 0)
            print("ctrl_points: ", self.ctrl_points)
            print("grad_scale: ", self.grad_scale_real)
        elif self.search_cfg.model.type == 'FasterRCNN':
            torch.distributed.broadcast(self.grad_scale_rpn_real, 0)
            torch.distributed.broadcast(self.grad_scale_rcnn_real, 0)
            print("ctrl_points for rpn: ", self.ctrl_points[0][0:5])
            print("grad_scale for rpn: ", self.grad_scale_rpn_real)
            print("ctrl_points for rcnn: ", self.ctrl_points[0][5:10])
            print("grad_scale for rcnn: ", self.grad_scale_rcnn_real)

        # training
        print('Start Training......')
        if self.search_cfg.model.type == 'RetinaNet':
            model.module.bbox_head.ctrl_points = self.ctrl_points[0]
            model.module.bbox_head.grad_scale = self.grad_scale_real.item()
        elif self.search_cfg.model.type == 'FasterRCNN':
            model.module.rpn_head.ctrl_points = self.ctrl_points[0][0:5]
            model.module.roi_head.bbox_head.ctrl_points = self.ctrl_points[0][5:10]
            model.module.rpn_head.grad_scale = self.grad_scale_rpn_real.item()
            model.module.roi_head.bbox_head.grad_scale = self.grad_scale_rcnn_real.item()

        # train and eval
        reward = train_iters(model, self.data_loader_train, self.data_loader_val, optimizer, 
                             lr_scheduler, torch.cuda.current_device(), self.args, self.search_cfg)
        
        print('reward this sample: ', reward)

        if self.search_cfg.model.type == 'RetinaNet':
            return torch.cat([self.theta.view(1, -1), (self.grad_scale.view(1, -1) + 1)/2], dim=1), torch.as_tensor(reward).to(self.theta).view(1, 1)
        elif self.search_cfg.model.type == 'FasterRCNN':
            return torch.cat([self.theta.view(1, -1), (self.grad_scale_rpn.view(1, -1) + 1)/2, (self.grad_scale_rcnn.view(1, -1) + 1)/2], dim=1), torch.as_tensor(reward).to(self.theta).view(1, 1)
    
    def convert_theta_coordinate(self, theta_group):
        theta_group_ctrl = theta_group.clone().detach()
        points_num = theta_group_ctrl.shape[2] // 2 
        for i in range(points_num):
            if i == 0:
                theta_group_ctrl[:, :, i * 2:i * 2 + 2] = theta_group_ctrl[:, :, i * 2:i * 2 + 2] * 1
            else:
                theta_group_ctrl[:, :, i * 2:i * 2 + 2] = theta_group_ctrl[:, :, i * 2:i * 2 + 2] * (
                        1 - theta_group_ctrl[:, :, i * 2 - 2:i * 2]) + theta_group_ctrl[:, :, i * 2 - 2:i * 2]
        theta_group_ctrl = torch.cat([torch.zeros(theta_group_ctrl.shape[0], theta_group_ctrl.shape[1], 2).to(theta_group_ctrl),
                                      theta_group_ctrl,
                                      torch.ones(theta_group_ctrl.shape[0], theta_group_ctrl.shape[1], 2).to(theta_group_ctrl)],
                                      dim=2)
        return theta_group_ctrl


def train_iters(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device, args, cfg):
    count = 0
    loss_one_count = 0
    for epoch in range(cfg.total_epochs):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        iter_loader = iter(train_dataloader)
        inputs = next(iter_loader)
                
        for iters in metric_logger.log_every(range(len(train_dataloader)), 50):
            outputs = model.train_step(inputs, optimizer)
            loss = outputs['loss']
            
            # break when training fails
            break_flag = torch.tensor(0.0).cuda()
            if torch.abs(loss - 1.0) < 1e-6 or torch.isnan(loss):
                loss_one_count += 1
            if loss_one_count > 50:
                break_flag = torch.tensor(1.0).cuda()
            torch.distributed.all_reduce(break_flag)
            if break_flag:
                return torch.tensor(0.0).cuda()

            lr_scheduler.step(count, epoch)
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            try:
                inputs = next(iter_loader)
            except StopIteration:
                inputs = None
            metric_logger.update(lr=optimizer.param_groups[0]['lr'],
                                 loss=outputs['log_vars']['loss'],)
    mAP = evaluate_model(model, val_dataloader)
    return mAP


@torch.no_grad()
def evaluate_model(model, dataloader):
    model.eval()
    with utils.NoPrint():
        from mmdet.apis import multi_gpu_test
        results = multi_gpu_test(model, dataloader, gpu_collect=True)

        mAP = torch.tensor(0.0).cuda()
        if torch.distributed.get_rank() == 0:
            if len(results) == len(dataloader.dataset):
                eval_res = dataloader.dataset.evaluate(results)
                if 'bbox_mAP' in eval_res:
                    mAP = torch.tensor(eval_res['bbox_mAP']).cuda()
                elif 'mAP' in eval_res:
                    mAP = torch.tensor(eval_res['mAP']).cuda()
        torch.distributed.all_reduce(mAP)
    return mAP


def argmax(mu_x, last_sigma, sigma, theta_groups, reward_groups, cfg, round_num):
    print('Updating ......')
    policy_optimizer = torch.optim.Adam([mu_x], lr=cfg.mu_lr)
    
    print('using warmup schedule for argmax')
    def lambda_func(epoch):
        update_steps = cfg.update_per_sample
        real_epoch = epoch % update_steps
        if real_epoch <= 30:
            return real_epoch/30
        else:
            return 1-((real_epoch-30)/(update_steps-30))
    
    policy_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(policy_optimizer, lr_lambda=lambda_func)
    
    sampler = TruncatedNormal(mu_x.sigmoid(), last_sigma, 0.0, 1.0)
    
    init_log_prob_groups = torch.zeros(theta_groups.shape, device=torch.cuda.current_device())
    init_log_prob_groups = sampler.log_prob(theta_groups).clone().detach()
    reward_groups = reward_groups.view(-1)
    
    for it in range(cfg.update_per_sample):
        sampler = TruncatedNormal(mu_x.sigmoid(), sigma, 0.0, 1.0)
        
        log_prob_groups = torch.zeros(theta_groups.shape, device=torch.cuda.current_device())
        log_prob_groups = sampler.log_prob(theta_groups)
        
        discount = (log_prob_groups - init_log_prob_groups.detach()).sum(dim=-1).exp()
        discount_clip = (log_prob_groups - init_log_prob_groups.detach()).clamp(math.log(1.0 - cfg.clip_eps), math.log(1.0 + cfg.clip_eps)).sum(dim=-1).exp()
        policy_loss = torch.min(discount * reward_groups, discount_clip * reward_groups)
        policy_loss = -1.0 * policy_loss.mean(dim=0)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        policy_lr_scheduler.step()

    print('Argmax update finish.')


def save_result(save_mu_list, save_sigma_list, save_theta_list, rounds, rewards, output_dir):
    if torch.distributed.get_rank() == 0:
        # save mu for each time, this is the mu that can directly transfer into function points
        save_mu_array = np.array(save_mu_list)
        save_np_path = os.path.join(output_dir, 'mu.npy')
        np.save(save_np_path, save_mu_array)

        # save sigma for each time
        save_mu_array = np.array(save_sigma_list)
        save_np_path = os.path.join(output_dir, 'sigma.npy')
        np.save(save_np_path, save_mu_array)

        # save sample funcs for each time
        save_theta_array = np.array(save_theta_list)
        save_theta_path = os.path.join(output_dir, 'sample_theta.npy')
        np.save(save_theta_path, save_theta_array)

        # # save pic for each round
        # plt.figure()
        # plt.plot(rounds, rewards)
        # save_path = os.path.join(output_dir, 'search.png')
        # plt.savefig(save_path)


def main(args):
    cfg = Config.fromfile(args.cfg_path)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(args)
    print(cfg)

    auto_loss = Parametrized_AP_Loss(args=args, cfg=cfg)
    auto_loss.search()

    plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameterized AP Loss search process', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
