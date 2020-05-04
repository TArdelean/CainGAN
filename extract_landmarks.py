import os
import pickle

import cv2
import face_alignment

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='vox2selection/mp4', help='Path to mp4 root')
parser.add_argument('--output_path', type=str, default='vox2selection/land', help='Path to output')
parser.add_argument('--device', type=str, default='cpu', help='CUDA or CPU')
args = parser.parse_args()

device = args.device
data_root = args.data_root
output_path = args.output_path

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)


def get_landmarks(frame):
    landmarks = fa.get_landmarks(frame)
    if landmarks is not None:
        landmarks = landmarks[0]
    return landmarks


def save_video_landmarks(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lands = []
    for i in tqdm(range(length)):
        ret, frame = cap.read()
        land = get_landmarks(frame)
        if land is None:
            print("No landmark")
            continue
        lands.append((i, land))

    cap.release()

    with open(os.path.join(out_dir, 'points.pickle'), 'wb') as handle:
        pickle.dump(lands, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_landmarks(path):
    for person_id in os.listdir(path):
        for video_id in os.listdir(os.path.join(path, person_id)):
            for video in os.listdir(os.path.join(path, person_id, video_id)):
                utt_path = os.path.join(path, person_id, video_id, video)
                out_utt_dir = os.path.join(output_path, person_id, video_id, video[:-4])
                if os.path.exists(out_utt_dir):
                    continue

                print(f"Started {utt_path}")
                os.makedirs(out_utt_dir)
                save_video_landmarks(utt_path, out_utt_dir)
                print(f"Finished {utt_path}")


if __name__ == '__main__':
    compute_landmarks(data_root)
