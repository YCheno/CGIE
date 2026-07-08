import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.GramLoss import VGGLoss

logger = logging.getLogger('base')


class SPSRModel(BaseModel):
    def __init__(self, opt):
        super(SPSRModel, self).__init__(opt)
        train_opt = opt['train']
        self.netG = networks.define_G(opt).to(self.device)
        if self.is_train:
            self.netG.train()
        self.load()

        if self.is_train:
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'l1_per':
                    self.cri_pix = nn.L1Loss().to(self.device)
                    self.vgg = VGGLoss()
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
            self.Branch_pretrain = train_opt['Branch_pretrain'] if train_opt['Branch_pretrain'] else 0
            self.Branch_init_iters = train_opt['Branch_init_iters'] if train_opt['Branch_init_iters'] else 1

            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(
                optim_params,
                lr=train_opt['lr_G'],
                weight_decay=wd_G,
                betas=(train_opt['beta1_G'], 0.999),
            )
            self.optimizers.append(self.optimizer_G)

            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR(
                            optimizer,
                            train_opt['lr_steps'],
                            train_opt['lr_gamma'],
                        )
                    )
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device)

        if need_HR:
            self.var_H = data['HR'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.output = self.netG(self.var_L)

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_g_pix = self.cri_pix(self.output, self.var_H)
            l_g_vgg = self.vgg(self.output, self.var_H)
            l_g_total = l_g_pix + l_g_vgg

            l_g_total.backward()
            self.optimizer_G.step()

            self.log_dict['l_g_pix'] = l_g_pix.item()
            self.log_dict['l_g_vgg'] = l_g_vgg.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.output = self.netG(self.var_L)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.output.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__,
                self.netG.module.__class__.__name__,
            )
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
