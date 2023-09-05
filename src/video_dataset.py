import torch
from typing import List, Dict
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
import os

N_HAND_KEYPOINTS = 21 * 3 * 2
N_FACE_KEYPOINTS = 468 * 3


class VideoDataset(Dataset):
    def __init__(self, root: os.path, directories: List[str], label_map: Dict[str, int], use_face: bool):
        self.root = root
        self.directories = directories
        self.label_map = label_map
        self.use_face = use_face

    def __len__(self):
        return len(self.directories)

    def __categorize(self, value) -> Tensor:
        categorized_label = [0] * len(self.label_map)
        categorized_label[self.label_map[value]] = 1
        return Tensor(categorized_label)

    def __getitem__(self, index) -> (Tensor, Tensor):
        keypoints_path = os.path.join(self.root, self.directories[index])
        label = os.listdir(keypoints_path)[0]
        frames = os.listdir(os.path.join(keypoints_path, label))
        frames.sort()
        if self.use_face:
            frame_window = np.empty((len(frames), N_HAND_KEYPOINTS + N_FACE_KEYPOINTS))
        else:
            frame_window = np.empty((len(frames), N_HAND_KEYPOINTS))
        for i in range(len(frames)):
            tmp = np.load(os.path.join(keypoints_path, label, frames[i]))
            if self.use_face:
                frame_window[i] = tmp[:][:N_FACE_KEYPOINTS+N_HAND_KEYPOINTS]
            else:
                frame_window[i] = tmp[:][:N_HAND_KEYPOINTS]
        return dict(
            label=self.__categorize(label),
            sequence=torch.from_numpy(frame_window).type(torch.FloatTensor)
        )
