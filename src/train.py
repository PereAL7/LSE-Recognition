import os

import torch
from playsound import playsound
from lse_predictor import LSEPredictor
from lse_predictor_v2 import LSEPredictorV2
from lse_predictor_v3 import LSEPredictorV3
from video_data_module import VideoDataModule
from sklearn.model_selection import train_test_split
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

HIDDEN_DIM = 256 * 2


def train(data_path: os.path, batch_size: int, label_map: dict, n_epochs: int, use_face: bool = True):
    paths = os.listdir(data_path)
    training_seq, testing_seq = train_test_split(paths, test_size=.3)
    input_dim = 21 * 3 * 2
    if use_face:
        input_dim += 468 * 3

    data_module = VideoDataModule(
        root=data_path,
        train_seq=training_seq,
        test_seq=testing_seq,
        batch_size=batch_size,
        label_map=label_map,
        use_face=use_face
    )

    # model = LSEPredictor(input_dim=input_dim,
    #                      hidden_dim=HIDDEN_DIM,
    #                      output_dim=len(label_map))
    # model = LSEPredictorV2(input_dim=input_dim,
    #                        hidden_dim=HIDDEN_DIM,
    #                        output_dim=len(label_map))
    model = LSEPredictorV3(input_dim=input_dim,
                           hidden_dim=HIDDEN_DIM,
                           output_dim=len(label_map))

    version = "arq_v3_4"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"best-checkpoint-{version}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("lightning_logs", name="my_model", version=version)

    trainer = Trainer(
        logger=logger,
        max_epochs=n_epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    checkpoints_path = "/home/pere/Documents/Sign-Language-Recognition/src/checkpoints"
    checkpoints = os.listdir(checkpoints_path)
    checkpoints.sort()

    torch.save(model,
               f"/home/pere/Documents/Sign-Language-Recognition/src/models/model_final_v"
               f"{len(os.listdir('/home/pere/Documents/Sign-Language-Recognition/src/models'))}")

    # trainer_model = LSEPredictor.load_from_checkpoint(
    #     os.path.join(checkpoints_path, checkpoints[-1]),
    #     input_dim=input_dim,
    #     hidden_dim=HIDDEN_DIM,
    #     output_dim=len(label_map)
    # )
    # trainer_model.freeze()
    # trainer.test(trainer_model, data_module)
    playsound('/home/pere/Downloads/pipe.mp3')


if __name__ == '__main__':
    train(data_path="/home/pere/Downloads/static_data",
          batch_size=30,
          label_map={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                     'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                     'X': 23, 'Y': 24, 'Z': 25},
          n_epochs=1500,
          use_face=True
          )

# {'A': 0, 'B': 1, 'C': 2}

# {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
# 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
# 'X': 23, 'Y': 24, 'Z': 25}

# {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'I': 6, 'K': 7, 'L': 8,
# 'M': 9, 'N': 10, 'O': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'U': 17}
