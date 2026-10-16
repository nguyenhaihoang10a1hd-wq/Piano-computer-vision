"""
Đàn piano ảo: camera + MediaPipe Hands + pygame.

Cài đặt:
    pip install opencv-python pygame numpy mediapipe==0.10.14

Chạy:  python main.py      (nhấn Q để thoát)
"""
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------- Cấu hình ----------
SOUND_DIR = r"C:\Users\DELL\Downloads\piano_sounds"    # thư mục chứa C4.wav ... B4.wav
NOTES = [("Do", "C4"), ("Re", "D4"), ("Mi", "E4"), ("Fa", "F4"),
         ("Sol", "G4"), ("La", "A4"), ("Si", "B4")]    # (tên hiển thị, tên file)
KEY_TOP_RATIO = 0.60             # vùng phím bắt đầu từ 60% chiều cao khung hình
FINGERTIPS = (4, 8, 12, 16, 20)  # chỉ số đầu ngón: cái -> út (theo MediaPipe)
RED = (0, 0, 255)                # màu viền phím (BGR)
HIGHLIGHT = (0, 200, 255)        # màu phím khi đang được chạm (BGR)
CAM_INDEX = 0                    # nếu không mở được camera, thử 1 hoặc 2
NUM_KEYS = len(NOTES)


def load_sounds():
    """Quét SOUND_DIR một lần (kể cả thư mục lồng nhau) và nạp 7 file âm thanh."""
    found = {}
    for root, _, files in os.walk(SOUND_DIR):
        for name in files:
            found.setdefault(name, os.path.join(root, name))

    sounds = []
    for _, note in NOTES:
        file_name = f"{note}.wav"
        if file_name not in found:
            sys.exit(f"Thiếu file {file_name} trong {SOUND_DIR}")
        sounds.append(pygame.mixer.Sound(found[file_name]))
    return sounds


def main():
    # Buffer nhỏ (512) để giảm độ trễ âm thanh; 16 kênh để phát được nhiều nốt cùng lúc
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.mixer.init()
    pygame.mixer.set_num_channels(16)
    sounds = load_sounds()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)   # CAP_DSHOW mở nhanh hơn trên Windows
    if not cap.isOpened():
        sys.exit("Không mở được camera. Thử đổi CAM_INDEX thành 1 hoặc 2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Lấy kích thước khung hình thực tế và tính vị trí các phím MỘT LẦN (không tính lại mỗi frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top = int(h * KEY_TOP_RATIO)
    key_w = w / NUM_KEYS
    edges = [int(i * key_w) for i in range(NUM_KEYS + 1)]          # biên trái/phải của từng phím
    label_pos = [(int(i * key_w + key_w / 2 - 25), h - 30) for i in range(NUM_KEYS)]

    last_key = {}   # (tay, đầu ngón) -> phím đang chạm ở frame trước (None nếu không chạm)

    with mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                  min_detection_confidence=0.6,
                                  min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)    # lật gương cho tự nhiên

            # Đặt writeable=False giúp MediaPipe xử lý nhanh hơn (không phải copy dữ liệu)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = hands.process(rgb)

            pressed = set()   # các phím đang bị chạm (để tô sáng)
            current = {}      # trạng thái của frame này, thay thế last_key ở cuối vòng lặp

            if res.multi_hand_landmarks:
                for lms, hand in zip(res.multi_hand_landmarks, res.multi_handedness):
                    side = hand.classification[0].label    # "Left" hoặc "Right"
                    for tip in FINGERTIPS:
                        pt = lms.landmark[tip]
                        x, y = int(pt.x * w), int(pt.y * h)
                        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

                        # Đầu ngón nằm trong vùng phím -> xác định phím (kẹp trong 0..6)
                        key = min(max(int(x // key_w), 0), NUM_KEYS - 1) if y >= top else None
                        idx = (side, tip)
                        current[idx] = key

                        if key is not None:
                            pressed.add(key)
                            # Chỉ phát khi đầu ngón VỪA đi vào phím (tránh lặp liên tục)
                            if last_key.get(idx) != key:
                                sounds[key].stop()    # phát lại từ đầu cho tiếng rõ
                                sounds[key].play()

            last_key = current    # ngón không còn được nhận diện sẽ tự bị loại khỏi dict

            # ---------- Vẽ giao diện ----------
            # Chỉ trộn màu trên vùng phím bị chạm (nhanh hơn copy + blend cả khung hình)
            for k in pressed:
                region = frame[top:h, edges[k]:edges[k + 1]]
                color = np.full_like(region, HIGHLIGHT)
                frame[top:h, edges[k]:edges[k + 1]] = cv2.addWeighted(region, 0.6, color, 0.4, 0)

            cv2.line(frame, (0, top), (w, top), RED, 4)             # cạnh trên của dàn phím
            for x in edges[1:-1]:
                cv2.line(frame, (x, top), (x, h), RED, 4)           # vạch ngăn giữa các phím
            for (name, _), pos in zip(NOTES, label_pos):
                cv2.putText(frame, name, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Virtual Piano", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()