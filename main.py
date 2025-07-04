import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import keys

# Load classes
classes = keys.keys()

# Load model once at start
model = load_model('QuickDrawCNN.h5')

# Constants
w, h = 700, 400
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(138, 43, 226), thickness=7)
handConStyle = mpDraw.DrawingSpec(color=(205, 16, 118), thickness=7)

def predict(data):
    # Preprocess image
    data = 255 - data
    data = data / 255.0
    predict_data = data.reshape(1, 28, 28, 1)

    # Predict
    pred = model.predict(predict_data)
    pred_class = np.argmax(pred)

    print(f"Predicted: {classes[pred_class]}")

    # Plot prediction
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title(f"Prediction: {classes[pred_class]}")
    plt.imshow(data, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Class Probabilities")
    plt.bar(range(10), pred.flatten())
    plt.xticks(range(10), classes, rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    draw_points = []
    is_drawing = False

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.flip(img_rgb, 1)
        img = cv2.flip(img, 1)
        result = hands.process(img_rgb)

        img[:] = (255, 255, 255)  # White canvas

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

                hand_local = [(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in handLms.landmark]

                linear_distance = np.linalg.norm(np.array(hand_local[4]) - np.array(hand_local[8]))
                if linear_distance <= 30:
                    if not is_drawing:
                        draw_points.append([-1])
                    is_drawing = True
                    px, py = int(hand_local[4][0]), int(hand_local[4][1])
                    if 195 < px < 505 and 45 < py < 355:
                        draw_points.append([px, py])
                        cv2.putText(img, f"{px}, {py}", (px - 70, py + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                    else:
                        draw_points.append([-1])
                else:
                    is_drawing = False

        # Draw path
        for p in range(len(draw_points) - 1):
            if draw_points[p] != [-1] and draw_points[p + 1] != [-1]:
                cv2.line(img, draw_points[p], draw_points[p + 1], (0, 0, 0), 7)

        # Draw UI
        cv2.rectangle(img, (195, 45), (505, 355), (0, 255, 0), 3)
        cv2.putText(img, 'Draw inside the rectangle.', (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, 'E: Confirm  W: Clear  Q: Quit', (5, 385), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Draw & Predict', img)

        key = cv2.waitKey(1)
        if key == ord('w'):
            draw_points.clear()
        elif key == ord('e'):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            drawn_img = gray[50:350, 200:500]
            resized = cv2.resize(drawn_img, (28, 28), interpolation=cv2.INTER_CUBIC)
            predict(resized)
            draw_points.clear()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
