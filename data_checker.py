import os
import cv2
import numpy as np
from typing import List, Dict
import PySimpleGUI as sg

DATA_PATH = r"C:\Users\Pere Alzamora\Desktop\output"
WIDTH = 680
HEIGHT = 480
N_KEYPOINTS_HANDS = 21
N_KEYPOINTS_FACE = 468
N_DIMENSIONS = 3
hand_graph = [
    [1, 5, 17],
    [2],
    [3],
    [4],
    [],
    [6, 9],
    [7],
    [8],
    [],
    [10, 13],
    [11],
    [12],
    [],
    [14, 17],
    [15],
    [16],
    [],
    [18],
    [19],
    [20],
    []
]

hand_colors = [
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0)
]


def get_character(path: os.path) -> str:
    return os.listdir(path)[0]


def get_frames(path: os.path) -> List[str]:
    return os.listdir(path)


def get_keypoints(path: os.path) -> Dict[str, List[Dict[str, np.float64]]]:
    keypoints = np.load(path)
    left_keypoints = []
    right_keypoints = []
    face_keypoints = []
    for i in range(N_KEYPOINTS_HANDS):
        left_keypoints.append({"x": keypoints[i * 3] * WIDTH,
                               "y": keypoints[(i * 3) + 1] * HEIGHT,
                               "z": keypoints[(i * 3) + 2]})
        right_keypoints.append({"x": keypoints[(i + 21) * 3] * WIDTH,
                                "y": keypoints[((i + 21) * 3) + 1] * HEIGHT,
                                "z": keypoints[((i + 21) * 3) + 2]})
    for i in range(N_KEYPOINTS_FACE):
        face_keypoints.append({"x": keypoints[(i+N_KEYPOINTS_HANDS * 2) * 3] * WIDTH,
                               "y": keypoints[((i+N_KEYPOINTS_HANDS * 2) * 3) + 1] * HEIGHT,
                               "z": keypoints[((i+N_KEYPOINTS_HANDS * 2) * 3) + 2]})
    return {"left": left_keypoints, "right": right_keypoints, "face": face_keypoints}


def draw_keypoints(hand_keypoints, character):
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    radius = 1
    circle_color = (0, 0, 255)  # BGR
    thickness = 2
    for i in range(len(hand_graph)):
        for line in hand_graph[i]:
            # Draw right hand connections
            start_point = (int(hand_keypoints["right"][i]["x"]), int(hand_keypoints["right"][i]["y"]))
            end_point = (int(hand_keypoints["right"][line]["x"]), int(hand_keypoints["right"][line]["y"]))
            if start_point != (0, 0) and end_point != (0, 0):
                cv2.line(image, start_point, end_point, hand_colors[i], thickness)

            # Draw left hand connections
            start_point = (int(hand_keypoints["left"][i]["x"]), int(hand_keypoints["left"][i]["y"]))
            end_point = (int(hand_keypoints["left"][line]["x"]), int(hand_keypoints["left"][line]["y"]))
            if start_point != (0, 0) and end_point != (0, 0):
                cv2.line(image, start_point, end_point, hand_colors[i], thickness)

    # Draw right hand points
    for landmark in hand_keypoints["right"]:
        center = (int(landmark['x']), int(landmark['y']))
        if center != (0, 0):
            cv2.circle(image, center, radius, circle_color,
                       thickness)

    # Draw left hand points
    for landmark in hand_keypoints["left"]:
        center = (int(landmark['x']), int(landmark['y']))
        if center != (0, 0):
            cv2.circle(image, center, radius, circle_color,
                       thickness)

    # Draw face points
    for landmark in hand_keypoints["face"]:
        center = (int(landmark['x']), int(landmark['y']))
        if center != (0, 0):
            cv2.circle(image, center, radius, circle_color,
                       thickness)

    font_x = 5
    font_y = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.75
    font_color = (0, 255, 0)
    font_thickness = 2

    image = cv2.flip(image, 1)
    cv2.putText(image,
                f"{character}",
                (font_x, font_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

    cv2.imshow("A", image)
    cv2.waitKey(1)


def main():
    # Get all subdirectories
    data_paths = os.listdir(DATA_PATH)

    # Sort the directories
    data_paths = [int(path) for path in data_paths]
    data_paths.sort()

    for data_path in data_paths:
        character = get_character(os.path.join(DATA_PATH, str(data_path)))
        frames = get_frames(os.path.join(DATA_PATH, str(data_path), character))
        for frame in frames:
            keypoints = get_keypoints(os.path.join(DATA_PATH, str(data_path), character, frame))
            draw_keypoints(keypoints, character)
        result = sg.popup_yes_no("Is it good?")
        if result != 'Yes':
            os.rename(os.path.join(DATA_PATH, str(data_path)), os.path.join(DATA_PATH, str(data_path) + "_DELETE"))
        else:
            os.rename(os.path.join(DATA_PATH, str(data_path)), os.path.join(DATA_PATH, str(data_path) + "_KEEP"))


if __name__ == '__main__':
    main()
