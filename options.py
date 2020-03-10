import os

from parsable_options import ParsableOptions

opt = None


class Options(ParsableOptions):

    def __init__(self, suppress_parse=False):
        super().__init__(suppress_parse)

    # noinspection PyAttributeOutsideInit
    def initialize(self):
        self.device = 'cuda:0'
        self.cuda_visible = '1,2,5'
        self.image_size = (224, 224)
        self.final_output_size = None
        self.K = 8
        self.data_root = 'vox2selection/mp4'
        self.landmark_root = 'vox2selection/land'
        self.checkpoint_path = 'checkpoints/'
        self.max_epoch = 30
        self.batch_size = 2
        self.data_load_threads = 2
        self.display_freq = 10
        self.save_freq = 100
        self.save_epoch_freq = 1
        self.experiment_name = 'demo'
        self.use_html = True
        self.resume = ''
        self.land_suffix = "points.pickle"

        self.D_steps_per_G = 2
        self.D_ID_start = 0  # Epoch to start Identity Discriminator
        self.D_ID_grow = 10  # Numbers of epochs until Identity reaches maximum importance
        self.D_ID_lambda = 1  # Coefficient of Identity Discriminator

        self.VGG19_WEIGHT = 1e1
        self.FM_WEIGHT = 1e1
        self.LEARNING_RATE_EG = 4e-4
        self.LEARNING_RATE_D = 1e-4

        self.test_output_dir = '../tests/demo/images'
        self.test_input_meta = '../tests/meta.txt'

    # noinspection PyAttributeOutsideInit
    def proc(self):
        self.weight_path = os.path.join(self.checkpoint_path, self.experiment_name, 'weight')
        self.img_dir = os.path.join(self.checkpoint_path, self.experiment_name, 'img_dir')
        self.web_dir = os.path.join(self.checkpoint_path, self.experiment_name, 'web_dir')
        if self.final_output_size is None:
            self.final_output_size = (224, 224)
        if self.resume == '':
            self.resume = None
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)


def get_options():
    global opt
    if opt is None:
        opt = Options()
    return opt
