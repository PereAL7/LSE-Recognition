import cv2
import mediapipe as mp
import numpy as np
import os
import PySimpleGUI as sg

from utils import model_detection, draw_landmarks, extract_keypoints

# [GLOBAL VARIABLES]
N_FRAMES = 30
WINDOW_NAME = 'Webcam'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
N_KEYPOINTS_HANDS = 21
N_KEYPOINTS_FACE = 468
N_DIMENSIONS = 3


def display_countdown(character, n_sequence, n_sequences, image):
    image = cv2.resize(image, (640, 480))

    font_x = 5
    font_y = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.75
    font_color = (0, 0, 0)
    font_thickness = 2

    if character != '0':
        cv2.putText(image,
                    f"Secuencia {n_sequence + 1} de {n_sequences}",
                    (font_x, font_y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(image,
                    f"para la letra \"{character}\"",
                    (font_x, font_y + 35), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness, cv2.LINE_AA)
    cv2.imshow(WINDOW_NAME, image)


def save_result(results, character, frame_count, n_sequence, data_path):
    # Get keypoints
    keypoints = extract_keypoints(results)

    path = os.path.join(data_path, str(n_sequence), character)
    if not os.path.exists(path):
        os.makedirs(path)

    # Save data
    np.save(f"{path}/{frame_count:03d}", keypoints)


def save_results(results: list, character: str, idx: int, data_path: str):
    for i in range(len(results)):
        save_result(results[i], character, i, idx, data_path)


def main():
    characters = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ0"]
    layout = [
        [
            sg.Text("¿Dónde le gustaría guardar los datos?"),
            sg.In(size=(25, 1), enable_events=True, key="dir", disabled=True),
            sg.FolderBrowse(),
        ],
        [
            sg.Text("Nº de repeticiones por signo"),
            sg.InputText(size=(25, 1), enable_events=True, key="n_seq")
        ],
        [
            sg.Text("Letras a grabar (introducir sin comas)"),
            sg.InputText(size=(25, 1), enable_events=True, key="todo_characters")
        ],
        [
            sg.Button("Empezar"), sg.Button("Cancelar")
        ]
    ]

    window = sg.Window("Programa de Grabación de Lenguaje de Signos", layout)
    n_sequences = -1
    save_dir = None
    while True:
        event, values = window.read()
        if event == "Empezar":
            try:
                n_sequences = int(values['n_seq'])
            except ValueError:
                sg.popup("Valor no válido introducido en un campo numérico.", title="ERROR", )
                continue

            if values['dir'] == "":
                sg.popup("Por favor introduzca la dirección donde quiera guardar los archivos.", title="ERROR")
                continue
            if values['todo_characters']:
                characters = list(set(values['todo_characters'].upper()))
                characters.sort()
                characters.append('0')
            save_dir = values['dir']
            sg.popup("Perfecto! En cuanto cierre esta ventana van a empezar la grabaciones.\n"
                     "\n"
                     "1) Entre grabación y grabación se mostrará cómo se tiene que hacer el gesto.\n"
                     "2) Para empezar una grabación pulse la barra espaciadora.\n"
                     "3) Tiene poco tiempo para realizar cada gesto, así que cuidado!",
                     title="Instrucciones")
            break
        if event == "Cancelar" or event == sg.WINDOW_CLOSED:
            break
    window.close()

    if n_sequences < 1 or save_dir is None:
        return

    images = [cv2.imread(f"./abecedario/{character}.png")
              for character in characters]
    image_iter = iter(images)
    image = next(image_iter)

    video_capture = cv2.VideoCapture(0)

    frame_count = N_FRAMES
    frames_data = None

    char_iter = iter(characters)
    character = next(char_iter)
    n_sequence = 0
    count = 0

    with mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as pose_detector:
        while True:
            # Advance characters
            if n_sequence == n_sequences:
                n_sequence = 0
                character = next(char_iter)
                image = next(image_iter)

            # Record hand
            if frame_count != N_FRAMES:

                success, frame = video_capture.read()
                if not success:
                    exit(0)

                frame, results = model_detection(frame, pose_detector)

                draw_landmarks(frame, results)

                frame = cv2.flip(frame, 1)

                frames_data["keypoints"].append(results)

                cv2.imshow(WINDOW_NAME, frame)
                frame_count += 1

            # Show tutorial
            if frame_count == N_FRAMES:
                display_countdown(character, n_sequence, n_sequences, image)

            key = cv2.waitKey(1)

            # Start recording
            if key & 0xFF == ord(' ') and frame_count == N_FRAMES:
                # Save results
                if frames_data is not None:
                    save_results(results=frames_data["keypoints"], character=frames_data["character"], idx=count,
                                 data_path=save_dir)
                    count += 1

                if character == '0':
                    break

                frames_data = {"keypoints": [], "character": character}

                frame_count = 0
                n_sequence += 1

            # Close program
            if key & 0xFF == ord('q') or \
                    cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
