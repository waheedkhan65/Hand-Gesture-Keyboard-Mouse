import cv2
import pyautogui
import numpy as np
import mediapipe as mp
import time

class HandTracker:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
            model_complexity=0  # Use lighter model for speed
        )
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                              self.mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                                              self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2))
        return img

    def find_position(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def fingers_up(self):
        """Detect which fingers are up (extended)"""
        fingers = []
        if len(self.lmList) != 21:
            return fingers

        # Thumb - special case (horizontal movement) - more lenient detection
        if self.lmList[4][1] > self.lmList[3][1]:  # Right hand
            fingers.append(1 if self.lmList[4][1] > self.lmList[3][1] + 20 else 0)
        else:  # Left hand
            fingers.append(1 if self.lmList[4][1] < self.lmList[3][1] - 20 else 0)

        # Four fingers - check if tip is above PIP joint
        for tip in [8, 12, 16, 20]:
            # Finger is up if tip is above (smaller y) than PIP joint
            pip = tip - 2
            fingers.append(1 if self.lmList[tip][2] < self.lmList[pip][2] else 0)

        return fingers

    def find_distance(self, p1, p2):
        """Calculate distance between two landmarks"""
        if len(self.lmList) < max(p1, p2) + 1:
            return float('inf'), [0, 0, 0, 0]
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        length = np.hypot(x2 - x1, y2 - y1)
        return length, [x1, y1, x2, y2]


def run_virtual_mouse():
    # Setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Try to set higher FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for less lag
    
    detector = HandTracker(maxHands=1, detectionCon=0.5, trackCon=0.5)
    screen_w, screen_h = pyautogui.size()
    
    # Mouse smoothing
    prev_x, prev_y = 0, 0
    smooth_factor = 3  # Lower = smoother but faster response
    
    # Scroll tracking
    scroll_base_y = 0
    in_scroll_mode = False
    last_scroll_time = 0
    scroll_sensitivity = 2  # Increase for faster scrolling
    
    # Click states
    left_click_ready = True
    right_click_ready = True
    
    # FPS calculation
    prev_time = 0
    
    # Frame skip for performance
    frame_count = 0
    process_every_n_frames = 1  # Process every frame for smoothness
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        if frame_count % process_every_n_frames == 0:
            frame = detector.find_hands(frame, draw=True)
            lm_list = detector.find_position(frame, draw=False)
            
            if lm_list:
                # Get finger states
                fingers = detector.fingers_up()
                
                # Calculate total fingers up
                total_fingers_up = sum(fingers)
                
                # SCROLL MODE - Check if hand is in a fist (all fingers down or only thumb up)
                wrist_y = lm_list[0][2]
                middle_finger_base_y = lm_list[9][2]
                
                # Better fist detection: check if fingertips are close to palm
                is_fist = True
                if len(fingers) == 5:
                    # Check if fingers 1-4 are down (index, middle, ring, pinky)
                    for i in range(1, 5):
                        if fingers[i] == 1:  # If any finger is up
                            is_fist = False
                            break
                
                # Alternative fist check - fingertips below middle of hand
                if is_fist or (total_fingers_up <= 1 and lm_list[8][2] > lm_list[5][2]):  
                    if not in_scroll_mode:
                        in_scroll_mode = True
                        scroll_base_y = wrist_y
                        cv2.putText(frame, "SCROLL MODE ACTIVE", (200, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    else:
                        # Continuous scrolling based on position
                        current_time = time.time()
                        if current_time - last_scroll_time > 0.05:  # Limit scroll rate
                            scroll_delta = (scroll_base_y - wrist_y) * scroll_sensitivity / 10
                            
                            if abs(scroll_delta) > 0.5:  # Dead zone
                                pyautogui.scroll(int(scroll_delta))
                                last_scroll_time = current_time
                                
                                if scroll_delta > 0:
                                    cv2.putText(frame, ">>> SCROLL UP >>>", (200, 100), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                else:
                                    cv2.putText(frame, "<<< SCROLL DOWN <<<", (200, 100), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Visual indicator for scroll mode
                    cv2.rectangle(frame, (180, 30), (460, 120), (0, 255, 255), 2)
                    
                else:
                    in_scroll_mode = False
                    scroll_base_y = 0
                    
                    # MOUSE MOVEMENT MODE - Index finger up
                    if fingers[1] == 1:  # Index finger is up
                        # Get index finger tip position
                        x_index, y_index = lm_list[8][1], lm_list[8][2]
                        
                        # Draw tracking circle
                        cv2.circle(frame, (x_index, y_index), 12, (255, 0, 255), cv2.FILLED)
                        
                        # Create movement box for better control
                        cv2.rectangle(frame, (80, 80), (560, 400), (255, 255, 0), 2)
                        
                        # Map to screen with adjusted boundaries
                        screen_x = np.interp(x_index, (80, 560), (0, screen_w))
                        screen_y = np.interp(y_index, (80, 400), (0, screen_h))
                        
                        # Smooth movement
                        curr_x = prev_x + (screen_x - prev_x) / smooth_factor
                        curr_y = prev_y + (screen_y - prev_y) / smooth_factor
                        
                        # Move mouse
                        pyautogui.moveTo(curr_x, curr_y, duration=0)
                        prev_x, prev_y = curr_x, curr_y
                        
                        # LEFT CLICK - Thumb & Index touch (more lenient detection)
                        thumb_index_dist, coords = detector.find_distance(4, 8)
                        
                        # Visual feedback for proximity
                        if thumb_index_dist < 60:  # Show when fingers are getting close
                            proximity_color = (0, int(255 * (1 - thumb_index_dist/60)), int(255 * thumb_index_dist/60))
                            cv2.line(frame, (coords[0], coords[1]), (coords[2], coords[3]), proximity_color, 2)
                            cv2.circle(frame, (coords[0], coords[1]), 8, proximity_color, cv2.FILLED)
                            cv2.circle(frame, (coords[2], coords[3]), 8, proximity_color, cv2.FILLED)
                        
                        if thumb_index_dist < 40:  # Increased threshold for easier clicking
                            # Big visual feedback when touching
                            cv2.circle(frame, (coords[0], coords[1]), 20, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, (coords[2], coords[3]), 20, (0, 255, 0), cv2.FILLED)
                            cv2.putText(frame, "LEFT CLICK!", (50, 70), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            
                            if left_click_ready:
                                pyautogui.click()
                                left_click_ready = False
                        else:
                            left_click_ready = True
                        
                        # RIGHT CLICK - Thumb & Middle finger touch (check only when middle is up)
                        if fingers[2] == 1:  # Middle finger is up
                            thumb_middle_dist, coords2 = detector.find_distance(4, 12)
                            
                            # Visual feedback for proximity
                            if thumb_middle_dist < 60:  # Show when fingers are getting close
                                proximity_color = (int(255 * thumb_middle_dist/60), 0, int(255 * (1 - thumb_middle_dist/60)))
                                cv2.line(frame, (coords2[0], coords2[1]), (coords2[2], coords2[3]), proximity_color, 2)
                                cv2.circle(frame, (coords2[0], coords2[1]), 8, proximity_color, cv2.FILLED)
                                cv2.circle(frame, (coords2[2], coords2[3]), 8, proximity_color, cv2.FILLED)
                            
                            if thumb_middle_dist < 40:  # Increased threshold for easier clicking
                                # Big visual feedback when touching
                                cv2.circle(frame, (coords2[0], coords2[1]), 20, (0, 0, 255), cv2.FILLED)
                                cv2.circle(frame, (coords2[2], coords2[3]), 20, (0, 0, 255), cv2.FILLED)
                                cv2.putText(frame, "RIGHT CLICK!", (50, 120), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                
                                if right_click_ready:
                                    pyautogui.rightClick()
                                    right_click_ready = False
                            else:
                                right_click_ready = True
                
                # Show finger status
                status_text = "Fingers: " + "".join(['↑' if f else '↓' for f in fingers])
                cv2.putText(frame, status_text, (10, 460), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Calculate and show FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Q to quit", (550, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Virtual Mouse - Gesture Control", frame)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset calibration
            prev_x, prev_y = 0, 0
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configure PyAutoGUI for better performance
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0  # Remove built-in delay
    
    print("\n" + "="*50)
    print("VIRTUAL MOUSE - GESTURE CONTROL")
    print("="*50)
    print("\nGESTURES:")
    print("├─ INDEX FINGER UP → Move mouse pointer")
    print("├─ PINCH THUMB + INDEX → Left click")
    print("├─ PINCH THUMB + MIDDLE → Right click (raise middle finger first)")
    print("├─ CLOSED FIST + MOVE → Scroll up/down")
    print("└─ PRESS 'Q' → Quit program")
    print("\nTIPS:")
    print("• Keep hand steady for accurate control")
    print("• Make clear gestures for best recognition")
    print("• Move hand within the yellow box area")
    print("\nStarting in 2 seconds...")
    time.sleep(2)
    
    run_virtual_mouse()