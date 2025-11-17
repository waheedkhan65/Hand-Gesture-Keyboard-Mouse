import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, maxHands=1, detection_conf=0.7, track_conf=0.7):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=maxHands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        return frame

    def find_position(self, frame):
        lmList = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))
        return lmList
