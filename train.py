import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from cain_helper import CainTrainer
from dataset.vid_dataset import VidDataset
from options import get_options, Options
from utils import util
from utils.visualizer import Visualizer
from collections import OrderedDict

opt: Options = get_options()
if opt.device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_visible
dataset = VidDataset(opt)
dataLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.data_load_threads)

last_display_step = len(dataLoader) - len(dataLoader) % opt.display_freq
print(f"Last display step: {last_display_step}")
visualizer = Visualizer(last_display_step)

trainer = CainTrainer(opt)
lossesG, lossesD, current_epoch, i_batch_current = trainer.init_networks()


def train():
    batch_start = datetime.now()
    global current_epoch, i_batch_current
    global lossesG, lossesD
    staring_point = i_batch_current

    print(f"Start training from epoch {current_epoch}")
    torch.set_grad_enabled(True)
    for epoch in range(current_epoch, opt.max_epoch):
        epoch_G_loss = []
        epoch_D_loss = []

        epoch_start = datetime.now()
        for i_batch, (frames, marks, i) in enumerate(dataLoader, start=staring_point):
            staring_point = 0
            if i_batch > len(dataLoader):
                break
            x = frames[:, 0]
            g_y = marks[:, 0]
            f_lm = torch.cat((frames[:, 1:], marks[:, 1:]), dim=2)

            data = f_lm.view(f_lm.shape[0], -1, f_lm.shape[-2], f_lm.shape[-1]), g_y, x

            if i_batch % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(*data, epoch=epoch)

            trainer.run_discriminator_one_step(*data)

            if i_batch % opt.display_freq == 0:
                batch_end = datetime.now()
                avg_time = (batch_end - batch_start) / (opt.display_freq if i_batch != 0 else 1)

                print('\n\navg batch time for batch size of', x.shape[0], ':', avg_time)
                batch_start = datetime.now()

                print('[%d/%d][%d/%d]' % (epoch, opt.max_epoch, i_batch, len(dataLoader)))
                latest_losses = trainer.get_latest_losses()
                print(latest_losses)

                disp_x = x[0]
                disp_x_hat = trainer.get_latest_generated()[0]
                disp_y = g_y[0]
                visuals = OrderedDict(
                    [('real_frame', util.tensor2im(frames[0, 1].data)),
                     ('input_landmark', util.tensor2im(disp_y.data)),
                     ('ground_face', util.tensor2im(disp_x.data)),
                     ('synth_face', util.tensor2im(disp_x_hat.data)),
                     ])
                visualizer.save_current_results(visuals, epoch, i_batch)

                lossD = sum((value for (key, value) in latest_losses[1].items()))
                lossEG = sum((value for (key, value) in latest_losses[0].items()))
                lossesD.append(lossD)
                lossesG.append(lossEG)
                epoch_D_loss.append(lossD)
                epoch_G_loss.append(lossEG)

            if i_batch % opt.save_freq == 0:
                print('Saving latest...')
                trainer.save_networks(lossesG, lossesD, epoch, i_batch, 'latest')

        print(f'Finished epoch {epoch} in {datetime.now() - epoch_start}\t|\t'
              f'Mean G loss {np.mean(epoch_G_loss)}\t|\t'
              f'Mean D loss {np.mean(epoch_D_loss)}\t|\t')
        if (epoch+1) % opt.save_epoch_freq == 0:
            print(f'saving...')
            trainer.save_networks(lossesG, lossesD, epoch + 1, 0)


def main():
    train()


if __name__ == '__main__':
    main()
