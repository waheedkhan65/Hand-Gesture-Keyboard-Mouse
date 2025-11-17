import cv2
import pyautogui
import time
import math

# Try to import hand tracking module
try:
    from hand_tracking import HandTracker
except ImportError:
    print("HandTracker module not found. Using MediaPipe directly.")
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    class HandTracker:
        def __init__(self):
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        def find_hands(self, img, draw=True):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(img_rgb)
            
            if self.results.multi_hand_landmarks and draw:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            return img
        
        def find_position(self, img):
            lm_list = []
            if self.results.multi_hand_landmarks:
                # Get the first hand detected
                my_hand = self.results.multi_hand_landmarks[0]
                for id, lm in enumerate(my_hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
            return lm_list

# Create virtual keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
        ["Z", "X", "C", "V", "B", "N", "M", "<-", "SP"]]

# Store key coordinates for easier access
key_coords = {}

def draw_keyboard(frame):
    key_coords.clear()
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x, y = 60 * j + 20, 60 * i + 100
            key_coords[key] = (x, y, x+50, y+50)
            cv2.rectangle(frame, (x, y), (x+50, y+50), (255, 0, 0), 2)
            cv2.putText(frame, key, (x+15, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def get_key_at_position(x, y):
    for key, coords in key_coords.items():
        x1, y1, x2, y2 = coords
        if x1 < x < x2 and y1 < y < y2:
            return key
    return None

def find_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def run_virtual_keyboard():
    cap = cv2.VideoCapture(0)
    detector = HandTracker()
    
    # Variables for smoother operation
    last_key_pressed = None
    last_press_time = 0
    press_delay = 0.5  # Delay between key presses in seconds
    click_threshold = 30  # Distance threshold for thumb-index finger pinch
    
    print("=== VIRTUAL KEYBOARD STARTED ===")
    print("INSTRUCTIONS:")
    print("1. First, click on the application where you want to type (Notepad, Word, Browser, etc.)")
    print("2. Point at keys with your index finger")
    print("3. Pinch thumb and index finger together to type")
    print("4. Press 'q' on your physical keyboard to quit")
    print("5. Make sure the virtual keyboard window stays open")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame)
        
        # Get landmarks for hand
        lmList = detector.find_position(frame) if hasattr(detector, 'results') and detector.results.multi_hand_landmarks else []
        
        draw_keyboard(frame)
        
        current_key = None
        is_clicking = False
        
        if lmList and len(lmList) >= 21:  # Check if we have a full hand (21 landmarks)
            # Get index finger tip position for pointing (landmark 8)
            x, y = lmList[8][1], lmList[8][2]
            cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
            
            # Check if finger is inside a key box
            current_key = get_key_at_position(x, y)
            
            if current_key:
                x1, y1, x2, y2 = key_coords[current_key]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, current_key, (x1+15, y1+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check for click gesture (thumb and index finger pinch)
            # Get thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb_tip = (lmList[4][1], lmList[4][2])
            index_tip = (lmList[8][1], lmList[8][2])
            
            # Draw circles on thumb and index finger tips
            cv2.circle(frame, thumb_tip, 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, index_tip, 8, (255, 0, 255), cv2.FILLED)
            
            # Draw line between thumb and index finger
            cv2.line(frame, thumb_tip, index_tip, (255, 0, 255), 3)
            
            # Calculate distance between thumb and index finger
            distance = find_distance(thumb_tip, index_tip)
            
            # If distance is below threshold, consider it a click
            if distance < click_threshold:
                is_clicking = True
                cv2.circle(frame, thumb_tip, 8, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, index_tip, 8, (0, 255, 255), cv2.FILLED)
                cv2.line(frame, thumb_tip, index_tip, (0, 255, 255), 3)
                
                # Display click status
                cv2.putText(frame, "CLICK DETECTED - TYPING", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # If we're pointing at a key and clicking, press the key
        if current_key and is_clicking:
            current_time = time.time()
            if current_key != last_key_pressed or current_time - last_press_time > press_delay:
                # Type the key directly into the active application
                if current_key == "<-":
                    pyautogui.press('backspace')
                    print("Pressed: BACKSPACE")
                elif current_key == "SP":
                    pyautogui.press('space')
                    print("Pressed: SPACE")
                else:
                    pyautogui.write(current_key)
                    print(f"Typed: {current_key}")
                
                last_key_pressed = current_key
                last_press_time = current_time
                
                # Visual feedback for key press
                x1, y1, x2, y2 = key_coords[current_key]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, current_key, (x1+15, y1+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display instructions on the keyboard window
        cv2.putText(frame, "1. Click on your target app (Notepad, Browser, etc.)", (10, frame.shape[0] - 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "2. Point with index finger", (10, frame.shape[0] - 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "3. Pinch thumb & index to type", (10, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "4. Press 'q' to quit", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Virtual Keyboard - Type in any app!", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Virtual Keyboard closed.")

if __name__ == "__main__":
    run_virtual_keyboard()