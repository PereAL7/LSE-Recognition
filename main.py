import cv2
import torch
import mediapipe as mp

from utils import model_detection, paint_output, draw_landmarks, extract_keypoints, draw_line_between_landmarks, \
    draw_is_capturing, distance_between_landmarks
import utils as const

from lse_predictor import LSEPredictor
from lse_predictor_v2 import LSEPredictorV2
from lse_predictor_v3 import LSEPredictorV3

# [GLOBAL VARIABLES]
N_FRAMES = 30
WINDOW_NAME = 'Webcam'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
LABEL_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10,
             'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20,
             'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}


# {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'I': 6, 'K': 7, 'L': 8,
#  'M': 9, 'N': 10, 'O': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'U': 17}


def main():
    video_capture = cv2.VideoCapture(0)
    is_capturing = False
    sequence = torch.Tensor()
    output = None

    # trained_model = torch.load("src/models/model_final_v17")
    trained_model = LSEPredictorV3.load_from_checkpoint(
        "/home/pere/Documents/Sign-Language-Recognition/src/checkpoints/best-checkpoint-arq_v3_4.ckpt",
        input_dim=const.N_KEYPOINTS_HANDS * const.N_DIMENSIONS * 2 + const.N_KEYPOINTS_FACE * const.N_DIMENSIONS,
        hidden_dim=256*2,
        output_dim=len(LABEL_MAP)
    )

    trained_model.freeze()

    with mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose_detector:
        while True:
            success, frame = video_capture.read()
            if not success:
                exit(0)

            frame, results = model_detection(frame, pose_detector)

            draw_landmarks(frame, results)

            if not is_capturing:
                if results.left_hand_landmarks:
                    draw_line_between_landmarks(frame,
                                                results.left_hand_landmarks.landmark[4],
                                                results.left_hand_landmarks.landmark[8])
                    distance = distance_between_landmarks(
                        results.left_hand_landmarks.landmark[4],
                        results.left_hand_landmarks.landmark[8])
                    if distance > 0.2:
                        is_capturing = True
                        sequence = torch.Tensor()

            if is_capturing:
                keypoints = extract_keypoints(results)
                sequence = torch.cat((sequence, torch.from_numpy(keypoints).type(torch.FloatTensor).unsqueeze(0)), 0)
                if len(sequence) == 30:
                    output = trained_model(sequence.unsqueeze(dim=0))
                    is_capturing = False
                    sequence = torch.Tensor()

            draw_is_capturing(frame, is_capturing)

            frame = cv2.flip(frame, 1)

            if output is not None:
                paint_output(frame, output[0].tolist(), LABEL_MAP)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1)

            if key & 0xFF == ord(' '):
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
