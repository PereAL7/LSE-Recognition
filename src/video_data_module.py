import os

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from video_dataset import VideoDataset
from typing import List, Dict


class VideoDataModule(LightningDataModule):
    def __init__(self, train_seq: List[str], test_seq: List[str], batch_size: int, root: os.path,
                 label_map: Dict[str, int], use_face: bool = True):
        super().__init__()
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.batch_size = batch_size
        self.root = root
        self.label_map = label_map
        self.use_face = use_face

    def setup(self, stage: str) -> None:
        self.train_dataset = VideoDataset(root=self.root,
                                          directories=self.train_seq,
                                          label_map=self.label_map,
                                          use_face=self.use_face)
        self.test_dataset = VideoDataset(root=self.root,
                                         directories=self.test_seq,
                                         label_map=self.label_map,
                                         use_face=self.use_face)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )
