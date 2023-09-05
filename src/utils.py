import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
N_KEYPOINTS_HANDS = 21
N_KEYPOINTS_FACE = 468
N_DIMENSIONS = 3


def model_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def paint_output(image, predictions, label_map):
    for key, value in label_map.items():
        cv2.putText(image, f"{key}: {predictions[value] * 100:.2f}%", (50, 15 * (value + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1, cv2.LINE_AA)


def draw_landmarks(image, landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks.face_landmarks,
        connections=mp_holistic.FACEMESH_CONTOURS,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
    )


def draw_line_between_landmarks(image, a_point, b_point):
    a_point = (int(a_point.x * image.shape[1]), int(a_point.y * image.shape[0]))
    b_point = (int(b_point.x * image.shape[1]), int(b_point.y * image.shape[0]))
    color = (0, 0, 255)
    thickness = 2

    cv2.line(image, a_point, b_point, color, thickness)


def distance_between_landmarks(a_point, b_point):
    a_point = np.array((a_point.x, a_point.y))
    b_point = np.array((b_point.x, b_point.y))
    return np.linalg.norm(a_point - b_point)


def draw_is_capturing(image, flag):
    color = (0, 0, 255) if flag else (255, 0, 0)
    coords = (image.shape[1] - 20, image.shape[0] - 20)
    radius = 15
    thickness = -1
    cv2.circle(image, coords, radius, color, thickness)


def extract_keypoints(results):
    right_hand = np.zeros(N_KEYPOINTS_HANDS * N_DIMENSIONS)
    left_hand = np.zeros(N_KEYPOINTS_HANDS * N_DIMENSIONS)
    face = np.zeros(N_KEYPOINTS_FACE * N_DIMENSIONS)

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    # if results.left_hand_landmarks:
    #     left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    return np.concatenate([left_hand, right_hand, face])
