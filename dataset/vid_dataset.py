from torch.utils.data import Dataset
from torchvision import transforms
import torch
import PIL
import os

from dataset.video_extractor import select_frames, plot_landmarks, select_with_landmarks
from options import Options


class VidDataset(Dataset):
    def __init__(self, opt: Options):
        self.K = opt.K + 1
        self.data_root = opt.data_root
        self.device = opt.device
        self.land_suffix = opt.land_suffix
        self.transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
        ])
        self.landmark_root = opt.landmark_root
        self.video_paths, self.land_paths = self.retrieve_paths()

    def retrieve_paths(self):
        video_paths = []
        land_paths = []
        pres = 0
        for person_id in os.listdir(self.data_root):
            for video_id in os.listdir(os.path.join(self.data_root, person_id)):
                for video in os.listdir(os.path.join(self.data_root, person_id, video_id)):
                    # Precomputed landmark
                    exp_path = None
                    if self.landmark_root is not None:
                        exp_path = os.path.join(self.landmark_root, person_id, video_id, video[:-4], self.land_suffix)
                        if os.path.exists(exp_path):
                            pres += 1
                        else:
                            exp_path = None
                    land_paths.append(exp_path)
                    # Fallback
                    video_paths.append(os.path.join(self.data_root, person_id, video_id, video))
        print(f"Preprocessed landmarks {pres} out of {len(video_paths)}")
        return video_paths, land_paths

    def __getitem__(self, index):
        frames = marks = None
        while marks is None:
            if self.land_paths[index] is not None:
                frames, landmarks = select_with_landmarks(self.video_paths[index], pickle_file=self.land_paths[index],
                                                          K=self.K)
                marks = torch.stack([self.transform(plot_landmarks(frame, landmarks=land, device=self.device))
                                     for frame, land in zip(frames, landmarks)])
            else:
                print("computing lands")
                frames = select_frames(self.video_paths[index], self.K)
                marks = torch.stack([self.transform(plot_landmarks(frame, device=self.device))
                                     for frame in frames])
        return torch.stack([self.transform(PIL.Image.fromarray(frame.astype('uint8'), 'RGB')) for frame in frames]), \
               marks, index

    def __len__(self):
        return len(self.video_paths)
