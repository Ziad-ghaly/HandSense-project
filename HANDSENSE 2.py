import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def dist(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

def detect_gesture(lm):
    thumb, index, middle, ring, pinky = lm[4], lm[8], lm[12], lm[16], lm[20]
    thumb_mcp, index_mcp, middle_mcp, ring_mcp, pinky_mcp = lm[2], lm[5], lm[9], lm[13], lm[17]

    thumb_open = thumb.y < thumb_mcp.y
    index_open = index.y < index_mcp.y
    middle_open = middle.y < middle_mcp.y
    ring_open = ring.y < ring_mcp.y
    pinky_open = pinky.y < pinky_mcp.y

    # الحركات
    if thumb_open and index_open and middle_open and ring_open and pinky_open:
        if dist(thumb, index) > 0.1:
            return "Stop"
        return "Hello"
    if thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "Thumbs Up"
    if not thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "Fist"
    if index_open and middle_open and not ring_open and not pinky_open and thumb_open:
        return "Peace"
    if dist(thumb, index) < 0.05 and middle_open and ring_open and pinky_open:
        return "perfectooo"
    if thumb_open and index_open and not middle_open and not ring_open and pinky_open:
        return "love"
    if thumb_open and not index_open and not middle_open and ring_open and pinky_open:
        return "Call"
    return None

cap = cv2.VideoCapture(0)
last_gesture = None
stable_count = 0
STABLE_FRAMES = 5
p_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_lms.landmark)

            if gesture:
                cv2.putText(frame, gesture, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                stable_count = stable_count + 1 if gesture == last_gesture else 1
                last_gesture = gesture
                bar = min(100, int((stable_count / STABLE_FRAMES) * 100))
                cv2.rectangle(frame, (50, 120), (50 + bar*5, 140), (0, 255, 0), -1)
                cv2.putText(frame, f"Stability: {bar}%", (50, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        last_gesture = None
        stable_count = 0

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time - p_time != 0 else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {int(fps)}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Sign Language Interpreter", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
