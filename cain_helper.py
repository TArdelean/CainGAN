import os

import torch
from torch.nn import DataParallel

from network.CainGAN import CainGAN
from options import Options


class CainTrainer:
    def __init__(self, opt: Options):
        self.opt = opt
        self.model = CainGAN(opt)
        self.model = DataParallel(self.model)
        if opt.device != 'cpu':
            self.model = self.model.cuda()

        self.generated = None
        self.g_losses = None
        self.d_losses = None
        self.optimizer_EG, self.optimizer_D = self.model.module.create_optimizers(opt)

    def run_generator_one_step(self, *data, epoch):
        id_d_factor = 1 if self.opt.D_ID_grow == 0 else min(1, max(0, epoch - self.opt.D_ID_start) / self.opt.D_ID_grow)
        id_d_factor *= self.opt.D_ID_lambda

        self.optimizer_EG.zero_grad()
        g_losses, generated = self.model(*data, just_discriminator=False, infer=False)
        g_losses = {key: value.mean() for (key, value) in g_losses.items()}
        g_losses['Id_G'] *= id_d_factor
        g_losses['Land_Id_Feat'] *= id_d_factor
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_EG.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, *data):
        self.optimizer_D.zero_grad()
        d_losses, _ = self.model(*data, just_discriminator=True, infer=False)
        d_losses = {key: value.mean() for (key, value) in d_losses.items()}
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return [{key: value.item() for (key, value) in self.g_losses.items()},
                {key: value.item() for (key, value) in self.d_losses.items()}]

    def get_latest_generated(self):
        return self.generated

    def init_networks(self):
        """
        lossesG
        lossesD
        current_epoch, num_vid, i_batch_current
        """

        if not os.path.isfile(self.opt.weight_path + "-0") or self.opt.resume is None:
            print('Initiating new checkpoint...')
            self.save_networks([], [])
        current_epoch = self.opt.resume if self.opt.resume is not None else 0

        checkpoint = torch.load(self.opt.weight_path + "-" + str(current_epoch), map_location='cpu')
        self.model.load_state_dict(checkpoint['GAN_state_dict'])
        current_epoch = checkpoint['epoch']
        losses_g = checkpoint['lossesG']
        losses_d = checkpoint['lossesD']
        i_batch_current = checkpoint['i_batch']

        self.model.module.print_network()
        return losses_g, losses_d, current_epoch, i_batch_current

    def save_networks(self, losses_g, losses_d, epoch=0, i_batch=0, ep_name=None):
        if ep_name is None:
            ep_name = str(epoch)
        torch.save({
            'epoch': epoch,
            'lossesG': losses_g,
            'lossesD': losses_d,
            'GAN_state_dict': self.model.state_dict(),
            'i_batch': i_batch
        }, self.opt.weight_path + "-" + ep_name)
        print('...Done')


class CainTester:
    def __init__(self, opt):
        self.opt = opt
        self.model = CainGAN(opt)
        self.model = DataParallel(self.model)
        if opt.device != 'cpu':
            self.model = self.model.cuda()
        self.generated = None

    def infer(self, *data):
        self.generated = self.model(*data, infer=True)
        return self.generated

    def get_latest_generated(self):
        return self.generated

    def init_networks(self):
        if self.opt.resume is None:
            raise Exception("Need resume checkpoint for testing")
        current_epoch = self.opt.resume
        checkpoint = torch.load(self.opt.weight_path + "-" + str(current_epoch), map_location='cpu')
        self.model.load_state_dict(checkpoint['GAN_state_dict'])
        current_epoch = checkpoint['epoch']
        self.model.module.print_network()
        return current_epoch
