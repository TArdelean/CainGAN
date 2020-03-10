from typing import List, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
import torch
import PIL
import os

from dataset.video_extractor import select_frames, plot_landmarks, select_with_landmarks


class TestDataset(Dataset):
    def __init__(self, opt):
        self.K = opt.K + 1
        self.data_root = opt.data_root
        self.meta_path = opt.test_input_meta
        self.device = opt.device
        self.land_suffix = opt.land_suffix
        self.transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
        ])
        self.landmark_root = opt.landmark_root
        self.video_paths, self.land_paths, self.frame_is = self.retrieve_paths()

    def retrieve_paths(self):
        video_paths = []
        land_paths = []
        frame_is = []
        pres = 0

        with open(self.meta_path) as f:
            metas = f.readlines()
        metas = [x.strip().split(',') for x in metas]
        for (ident, frame_i) in metas:
            frame_i = int(frame_i)
            exp_path = None
            id_split = ident.split(';')
            vid_path = os.path.join(self.data_root, *id_split) + ".mp4"
            if self.landmark_root is not None:
                exp_path = os.path.join(self.landmark_root, *id_split, self.land_suffix)
                if os.path.exists(exp_path):
                    pres += 1
                else:
                    exp_path = None
            frame_is.append(frame_i)
            land_paths.append(exp_path)
            video_paths.append(vid_path)

        print(f"Preprocessed landmarks {pres} out of {len(video_paths)}")
        return video_paths, land_paths, frame_is

    def __getitem__(self, index):
        frames = marks = None
        while marks is None:
            if self.land_paths[index] is not None:
                frames, landmarks = select_with_landmarks(self.video_paths[index], pickle_file=self.land_paths[index],
                                                          K=self.K, mandatory=[self.frame_is[index]])
                marks = torch.stack([self.transform(plot_landmarks(frame, landmarks=land, device=self.device))
                                     for frame, land in zip(frames, landmarks)])
            else:
                print("computing lands")
                frames = select_frames(self.video_paths[index], self.K, [self.frame_is[index]])
                marks = torch.stack([self.transform(plot_landmarks(frame, device=self.device))
                                     for frame in frames])
        return torch.stack([self.transform(PIL.Image.fromarray(frame.astype('uint8'), 'RGB')) for frame in frames]), \
               marks, self.video_paths[index], self.frame_is[index]

    def __len__(self):
        return len(self.video_paths)


class FixedTestDataset(TestDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.frame_is, self.frame_g = [el[0] for el in self.frame_is], [el[1] for el in self.frame_is]

    def retrieve_paths(self):
        video_paths = []
        land_paths = []
        frame_is: List[Tuple[int, List[int]]] = []
        pres = 0

        with open(self.meta_path) as f:
            metas = f.readlines()
        metas = [x.strip().split(',') for x in metas]

        for (ident, frame_i) in metas:
            gt_frame = int(frame_i.split('=')[1])
            input_frames = [int(ind) for ind in frame_i.split("=")[0].split(";")]
            exp_path = None
            id_split = ident.split(';')
            vid_path = os.path.join(self.data_root, *id_split) + ".mp4"
            if self.landmark_root is not None:
                exp_path = os.path.join(self.landmark_root, *id_split, self.land_suffix)
                if os.path.exists(exp_path):
                    pres += 1
                else:
                    exp_path = None
            frame_is.append((gt_frame, input_frames))
            land_paths.append(exp_path)
            video_paths.append(vid_path)

        print(f"Preprocessed landmarks {pres} out of {len(video_paths)}")
        return video_paths, land_paths, frame_is

    def __getitem__(self, index):
        frames = marks = None
        while marks is None:
            if self.land_paths[index] is not None:
                frames, landmarks = select_with_landmarks(self.video_paths[index], pickle_file=self.land_paths[index],
                                                          K=self.K,
                                                          mandatory=[self.frame_is[index], *self.frame_g[index]])
                marks = torch.stack([self.transform(plot_landmarks(frame, landmarks=land, device=self.device))
                                     for frame, land in zip(frames, landmarks)])
            else:
                raise Exception("Computing landmarks not yet supported on Fixed Dataset")
        return torch.stack([self.transform(PIL.Image.fromarray(frame.astype('uint8'), 'RGB')) for frame in frames]), \
               marks, self.video_paths[index], self.frame_is[index]
