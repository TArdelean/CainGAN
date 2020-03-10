import os


from options import get_options

from utils import util, html


class Visualizer:
    def __init__(self, last_display_step):
        self.opt = get_options()
        self.img_dir = self.opt.img_dir
        self.last_step = last_display_step

    def get_img_path(self, epoch, step, label, i=None):
        if i is not None:
            return os.path.join(self.img_dir, 'epoch%.3d-%d_%s_%d.jpg' % (epoch, step, label, i))
        else:
            return os.path.join(self.img_dir, 'epoch%.3d-%d_%s.jpg' % (epoch, step, label))

    def save_current_results(self, visuals, epoch, step):
        for label, image_numpy in visuals.items():
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = self.get_img_path(epoch, step, label, i)
                    util.save_image(image_numpy[i], img_path)
            else:
                img_path = self.get_img_path(epoch, step, label)
                util.save_image(image_numpy, img_path)

        if self.opt.use_html:
            self.save_html_log(visuals, epoch, step)

    def save_html_log(self, visuals, epoch, step):

        # update website
        webpage = html.HTML(self.opt.web_dir, 'Experiment name = %s' % self.opt.experiment_name, refresh=30)
        for n in range(epoch, -1, -1):
            webpage.add_header('epoch [%d]' % n)
            ims = []
            txts = []
            links = []
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = self.get_img_path(n, self.last_step if n != epoch else step, label, i)
                        ims.append(img_path)
                        txts.append(label + str(i))
                        links.append(img_path)
                else:
                    img_path = self.get_img_path(n, self.last_step if n != epoch else step, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
            webpage.add_images(ims, txts, links)
        webpage.save()


