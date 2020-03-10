import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from cain_helper import CainTester
from dataset.test_dataset import TestDataset
from options import get_options
from utils import util


opt = get_options()
if opt.device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_visible
dataset = TestDataset(opt)
dataLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.data_load_threads)

tester = CainTester(opt)
current_epoch = tester.init_networks()


def save_test(synthesized, file_path, frame_i):
    im = util.tensor2im(synthesized.data)

    elements = file_path.split('/')
    ident = "-".join((elements[-3], elements[-2], elements[-1][:-4]))
    img_path = os.path.join(opt.test_output_dir, ident + "-" + str(frame_i) + ".png")
    util.save_image(im, img_path, opt.final_output_size)


def test():
    test_start = datetime.now()

    print(f"Testing checkpoint epoch {current_epoch}")
    for i_batch, (frames, marks, file_path, frame_i) in enumerate(dataLoader):
        x = frames[:, 0]
        g_y = marks[:, 0]
        f_lm = torch.cat((frames[:, 1:], marks[:, 1:]), dim=2)

        data = f_lm.view(f_lm.shape[0], -1, f_lm.shape[-2], f_lm.shape[-1]), g_y, x
        x_hat = tester.infer(*data)

        for i in range(x_hat.shape[0]):
            disp_x_hat = x_hat[i]
            save_test(disp_x_hat, file_path[i], frame_i[i].item())

        if (i_batch+1) % opt.display_freq == 0:
            time_elapsed = (datetime.now() - test_start)
            print('Batch', i_batch, '\t:\tTotal time elapsed ', time_elapsed)


def main():
    test()


if __name__ == '__main__':
    main()
