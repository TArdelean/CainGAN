import pickle

import PIL

import cv2
import face_alignment
from matplotlib import pyplot as plt
import numpy as np


def sample_frames(length, K, mandatory=None):
    if mandatory is None:
        mandatory = []
    options = np.setdiff1d(np.arange(length), np.array(mandatory), assume_unique=True)
    k = K - len(mandatory)
    if k <= len(options):
        sampled = np.random.choice(options, k, replace=False)
    else:
        print(f"Asking for {K} frames out of {length}; using replace mode")
        sampled = np.random.choice(options, K, replace=True)

    frame_idxs = mandatory + sampled.tolist()

    return frame_idxs


def mandatory_indexes(data, mandatory):
    if mandatory is None or mandatory == []:
        return mandatory
    existing_frames = [x[0] for x in data]

    def find_nearest(value):
        array = np.asarray(existing_frames)
        idx = (np.abs(array - value)).argmin()
        return idx
    return [find_nearest(x) for x in mandatory]


def select_with_landmarks(video_path, pickle_file=None, K=1, data=None, mandatory=None):
    """
    :param video_path: mp4 video file with real frames
    :param pickle_file: pickle file with landmark/pose annotations, used if data is None
    :param K: number of frames to pick at random
    :param data: landmark / pose annotations, if data is supplied pickle_file is ignored
    :param mandatory: List of frames that must be included
    :return: selected frames and landmarks
    """
    if data is None and pickle_file is None:
        raise Exception("At least data or pickle_file must be provided")
    if data is None:
        with open(pickle_file, 'rb') as handle:
            data = pickle.load(handle)
    mandatory = mandatory_indexes(data, mandatory)

    length = len(data)
    frame_idxs = sample_frames(length, K, mandatory)
    selected = [data[i] for i in frame_idxs]
    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_to_frame = np.full(length, -1)
    idx_to_frame[[x[0] for x in selected]] = np.arange(K)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames_list = np.empty((K, h, w, 3))
    landmark_list = np.empty((K, *data[0][1].shape))

    for i in range(length):
        ret, frame = cap.read()

        if idx_to_frame[i] != -1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            position = idx_to_frame[i]
            frames_list[position] = frame
            landmark_list[position] = selected[position][1]

    cap.release()

    return frames_list, landmark_list


def select_frames(video_path, K, mandatory=None):
    cap = cv2.VideoCapture(video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_idxs = sample_frames(length, K, mandatory=mandatory)
    idx_to_frame = np.full(length, -1)
    idx_to_frame[frame_idxs] = np.arange(K)

    frames_list = np.empty((K, h, w, 3))

    for i in range(length):
        ret, frame = cap.read()

        if idx_to_frame[i] != -1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list[idx_to_frame[i]] = frame

    cap.release()

    return frames_list


fa = None


def plot_landmarks(frame, landmarks=None, dpi=100, device='cpu'):
    global fa
    if landmarks is None:
        if fa is None:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
        landmarks = fa.get_landmarks(frame)[0]

    fig = plt.figure(figsize=(frame.shape[0] / dpi, frame.shape[1] / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(np.ones(frame.shape))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Draw Regions
    ax.plot(landmarks[0:17, 0], landmarks[0:17, 1], linestyle='-', color='green', lw=2)
    ax.plot(landmarks[17:22, 0], landmarks[17:22, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[22:27, 0], landmarks[22:27, 1], linestyle='-', color='orange', lw=2)
    ax.plot(landmarks[27:31, 0], landmarks[27:31, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[31:36, 0], landmarks[31:36, 1], linestyle='-', color='blue', lw=2)
    ax.plot(landmarks[36:42, 0], landmarks[36:42, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[42:48, 0], landmarks[42:48, 1], linestyle='-', color='red', lw=2)
    ax.plot(landmarks[48:60, 0], landmarks[48:60, 1], linestyle='-', color='purple', lw=2)

    fig.canvas.draw()
    data = PIL.Image.frombuffer('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb(), 'raw', 'RGB', 0, 1)
    plt.close(fig)
    return data
