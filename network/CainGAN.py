import torch
from torch import nn

import torch.nn.functional as F

from network import models
from network.blocks import ResBlockDown, ResnetStableBlock, ResBlockUp, SPADEResnetBlock
from network.models import weights_init
from network.models import MultiscaleDiscriminator
from network.loss import GANLoss, VGGLoss
from options import Options


class CainGAN(torch.nn.Module):

    def __init__(self, opt: Options = None):
        super(CainGAN, self).__init__()
        self.input = None
        self.loss = None
        self.opt: Options = opt
        self.targeted_embedder = TargetedEmbedder(ncf=64)
        self.generator = CainGenerator()
        self.land_discriminator, self.id_discriminator = (CainLandmarkDiscriminator(), CainIdentityDiscriminator())
        self.FloatTensor = torch.cuda.FloatTensor if opt.device != 'cpu' else torch.FloatTensor

        self.criterionGAN = GANLoss(tensor=self.FloatTensor)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = VGGLoss(opt.device != 'cpu')

    def name(self):
        return 'CainGAN'

    def encode_input(self, character, landmark, ground, infer):
        landmark.requires_grad_(not infer)
        character.requires_grad_(not infer)
        return character, landmark, ground

    def forward(self, character, landmark, ground, just_discriminator=False, infer=False):
        """
        :param landmark: Tensor B x in_l x W x H
        :param character: Tensor B x (K * in_c) x W x H
        :param ground: Tensor B x out_l x W x H
        :param just_discriminator:
        :param infer: True when testing
        :return: Synthesized image
        """

        character, landmark, ground = self.encode_input(character, landmark, ground, infer)

        if infer:
            with torch.no_grad():
                identity = self.targeted_embedder(character, landmark)
                generated = self.generator(landmark, identity)
                return generated
        elif just_discriminator:
            with torch.no_grad():
                identity = self.targeted_embedder(character, landmark)
                generated = self.generator(landmark, identity)
                generated = generated.detach()
                generated.requires_grad_()
            losses = {**self.discriminator_loss('Land', *self.land_discriminate(landmark, ground, generated)),
                      **self.discriminator_loss('Id', *self.id_discriminate(character[:, :3], ground, generated))}
        else:
            identity = self.targeted_embedder(character, landmark)
            generated = self.generator(landmark, identity)
            losses = self.generator_loss(character[:, :3], landmark, ground, generated,
                                         feat_loss=self.opt.FM_WEIGHT,
                                         vgg_loss=self.opt.VGG19_WEIGHT)

        return losses, generated

    def land_discriminate(self, landmark, ground, generated):
        res1 = self.land_discriminator(torch.cat((landmark, ground), dim=1))
        res2 = self.land_discriminator(torch.cat((landmark, generated), dim=1))
        return res1, res2

    def id_discriminate(self, character, ground, generated):
        res1 = self.id_discriminator(torch.cat((character, ground), dim=1))
        res2 = self.id_discriminator(torch.cat((character, generated), dim=1))
        return res1, res2

    def discriminator_loss(self, name, pred_real, pred_fake):
        return {
            f'{name}_D_Real': self.criterionGAN(pred_real, target_is_real=True, for_discriminator=True),
            f'{name}_D_Fake': self.criterionGAN(pred_fake, target_is_real=False, for_discriminator=True)
        }

    def generator_loss(self, character, landmark, ground, generated, feat_loss=0.01, vgg_loss=0.01):
        losses = {}

        def compute_feat_loss(pred_fake, pred_real):
            num_d = len(pred_fake)
            GAN_feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_d):
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_feat_loss += unweighted_loss * feat_loss / num_d
            return GAN_feat_loss

        land_real, land_fake = self.land_discriminate(landmark, ground, generated)
        losses['Land_G'] = self.criterionGAN(land_fake, True, for_discriminator=False)
        id_real, id_fake = self.id_discriminate(character, ground, generated)
        losses['Id_G'] = self.criterionGAN(id_fake, True, for_discriminator=False)
        if feat_loss is not None:
            losses['Land_G_Feat'] = compute_feat_loss(land_fake, land_real)
            losses['Land_Id_Feat'] = compute_feat_loss(id_fake, id_real)

        if vgg_loss is not None:
            losses['VGG'] = self.criterionVGG(generated, ground) * vgg_loss

        return losses

    def create_optimizers(self, opt: Options):
        eg_params = list(self.generator.parameters())
        eg_params += list(self.targeted_embedder.parameters())

        d_params = list(self.land_discriminator.parameters())
        d_params += list(self.id_discriminator.parameters())

        beta1, beta2 = 0, 0.9
        eg, d_lr = opt.LEARNING_RATE_EG, opt.LEARNING_RATE_D

        optimizer_eg = torch.optim.Adam(eg_params, lr=eg, betas=(beta1, beta2))
        optimizer_d = torch.optim.Adam(d_params, lr=d_lr, betas=(beta1, beta2))

        return optimizer_eg, optimizer_d

    def print_network(self):
        enc_np = models.print_network(self.targeted_embedder)
        gen_np = models.print_network(self.generator)
        disc_np = models.print_network(self.id_discriminator) + models.print_network(self.land_discriminator)

        print(f"Summary: ")
        print(f"Encoder - {enc_np}")
        print(f"Generator - {gen_np}")
        print(f"Discriminators - {disc_np}")
        print(f"Total - {enc_np + gen_np + disc_np}")

    def train(self, **kwargs):
        self.targeted_embedder.train(**kwargs)


class TargetedEmbedder(nn.Module):

    def __init__(self, n_char_down=2, n_char_up=0, n_char_same=4, in_c=6, in_t=3, ncf=64, mincf=16, maxcf=512, out_l=128):
        super(TargetedEmbedder, self).__init__()
        self.in_c = in_c
        self.in_l = in_c + in_t
        self.relu = nn.ReLU(inplace=False)
        model = []

        # in (B * K) x in_l x W x H
        model.append(ResnetStableBlock(self.in_l, ncf))
        last = ncf
        for i in range(n_char_down):
            model.append(ResBlockDown(last, min(maxcf, last * 2)))
            last = min(maxcf, last * 2)

        for i in range(n_char_up):
            model.append(ResBlockUp(last, max(mincf, last // 2)))
            last = max(mincf, last / 2)

        model.append(ResnetStableBlock(last, out_l))
        last = out_l
        for i in range(n_char_same):
            model.append(ResnetStableBlock(last, last))

        self.common = nn.Sequential(*model)
        self.to_id_maps = nn.Sequential(*([
            ResnetStableBlock(last, last)
        ] * (n_char_same // 2)))
        self.to_certainty = nn.Sequential(*([
            ResnetStableBlock(last, last)
        ] * (n_char_same // 2) + [nn.Sigmoid()]))
        self.apply(weights_init)
        pass

    def compute_mc(self, b_size, in_dim):
        common = self.common(in_dim)  # (B*K) x out_l x W x H
        id_maps = self.to_id_maps(common)
        certainty = self.to_certainty(common) + 0.01
        id_maps = id_maps.view(b_size, -1, *common.shape[-3:])  # B x K x out_l x W x H
        certainty = certainty.view(b_size, -1, *common.shape[-3:])  # B x K x out_l x W x H
        return id_maps, certainty

    # noinspection PyMethodMayBeStatic
    def combine(self, id_maps, certainty, strategy='ignore_certainty'):
        """
        Combines identity maps for K images weighting on certainty
        :param id_maps: Tensor B x K x out_l x W x H
        :param certainty: Tensor B x K x out_l x W x H
        :param strategy: [ignore_certainty, responsibility]
        :return: Combined identity: Tensor B x out_l x W x H
        """
        if strategy == 'ignore_certainty':
            return id_maps.mean(dim=1)
        weighted = id_maps * certainty
        weighted = weighted.sum(dim=1) / certainty.sum(dim=1)
        return weighted

    def forward(self, character, target_landmark):
        """
        :param character: Tensor B x (K * in_c) x W x H
        :param target_landmark: Tensor B x in_t x W x H
        :return: Encoded character: B x out_l x W x H
        """
        k = character.shape[1] // self.in_c
        in_dim = character.view(-1, self.in_c, character.shape[-2], character.shape[-1])  # (B*K) x in_c x W x H
        target = target_landmark.unsqueeze(1).expand(-1, k, -1, -1, -1).contiguous().\
            view(-1, *target_landmark.shape[-3:])  # (B*K) x in_t x W x H
        in_dim = torch.cat((in_dim, target), dim=1)  # (B*K) x in_l x W x H)
        id_maps, certainty = self.compute_mc(character.shape[0], in_dim)
        combined = self.combine(id_maps, certainty, "ignore_certainty")
        return combined


class CainGenerator(nn.Module):
    def __init__(self, in_l=3, out_c=128, ngf=64, n_blocks_same=3, out_l=3, maxgf=512, spectral=True):
        super(CainGenerator, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

        # in B x in_l x W x H
        self.resDown1 = ResBlockDown(in_l, ngf, conv_size=9, padding_size=4)
        self.in1 = nn.InstanceNorm2d(ngf, affine=True)

        self.resDown2 = ResBlockDown(ngf, ngf * 2, conv_size=3, padding_size=1)
        self.in2 = nn.InstanceNorm2d(ngf * 2, affine=True)

        self.resDown3 = ResBlockDown(ngf * 2, ngf * 4)
        self.in3 = nn.InstanceNorm2d(ngf * 4, affine=True)

        self.resDown4 = ResBlockDown(ngf * 4, ngf * 8)
        self.in4 = nn.InstanceNorm2d(ngf * 8, affine=True)

        last = min(maxgf, ngf * 8)
        same_res = [ResnetStableBlock(ngf * 8, last)]
        for i in range(n_blocks_same):
            same_res.append(ResnetStableBlock(last, last))
        self.same_res = nn.Sequential(*same_res)

        self.head = SPADEResnetBlock(last, 128 * out_l, out_c, spectral)
        last = 128 * out_l

        self.up_1 = SPADEResnetBlock(last, last // 2, out_c, spectral)
        self.up_2 = SPADEResnetBlock(last // 2, last // 4, out_c, spectral)
        self.up_3 = SPADEResnetBlock(last // 4, last // 8, out_c, spectral)
        self.up_4 = SPADEResnetBlock(last // 8, last // 16, out_c, spectral)
        last //= 16

        self.final = nn.Conv2d(last, out_l, kernel_size=3, padding=1)

    def forward(self, landmark, identity):
        """
        :param landmark: Tensor B x in_l x W x H
        :param identity: Tensor B x out_c x W' x H'
        :return: generated image: Tensor B x out_l x W x H
        """
        out = self.in1(self.resDown1(landmark))
        out = self.in2(self.resDown2(out))
        out = self.in3(self.resDown3(out))
        out = self.in4(self.resDown4(out))

        out = self.same_res(out)

        out = self.head(out, identity)
        out = self.up(out)
        out = self.up_1(out, identity)
        out = self.up(out)
        out = self.up_2(out, identity)
        out = self.up(out)
        out = self.up_3(out, identity)
        out = self.up(out)
        out = self.up_4(out, identity)

        out = self.final(out)
        out = F.sigmoid(out)

        return out


class CainLandmarkDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=2, getIntermFeat=True):
        super(CainLandmarkDiscriminator, self).__init__()
        self.model = MultiscaleDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, num_D, getIntermFeat)

    def forward(self, source):
        return self.model(source)


class CainIdentityDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=2, getIntermFeat=True):
        super(CainIdentityDiscriminator, self).__init__()
        self.model = MultiscaleDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, num_D, getIntermFeat)

    def forward(self, source):
        return self.model(source)
