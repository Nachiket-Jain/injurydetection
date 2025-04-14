import cv2
import mediapipe as mp
import math
import numpy as np

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle) if abs(angle) <= 180 else 360 - abs(angle)

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip and convert the frame for MediaPipe
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = result.pose_landmarks.landmark
        height, width, _ = frame.shape  # Get frame dimensions

        # Extract key points for angle calculations
        keypoints = {
            "shoulder_left": landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "shoulder_right": landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            "elbow_left": landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            "elbow_right": landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            "wrist_left": landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
            "wrist_right": landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
            "hip_left": landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            "hip_right": landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            "knee_left": landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            "knee_right": landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            "ankle_left": landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
            "ankle_right": landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
        }

        # Calculate angles
        angles = {
            "Left Arm": calculate_angle(keypoints["shoulder_left"], keypoints["elbow_left"], keypoints["wrist_left"]),
            "Right Arm": calculate_angle(keypoints["shoulder_right"], keypoints["elbow_right"], keypoints["wrist_right"]),
            "Left Leg": calculate_angle(keypoints["hip_left"], keypoints["knee_left"], keypoints["ankle_left"]),
            "Right Leg": calculate_angle(keypoints["hip_right"], keypoints["knee_right"], keypoints["ankle_right"]),
        }

        # Define color gradients for 3D effect
        def gradient_color(value, min_val=50, max_val=180):
            ratio = (value - min_val) / (max_val - min_val)
            return (int(255 * (1 - ratio)), int(255 * ratio), 100)

        # Draw 3D effect lines
        for (joint1, joint2, joint3), angle_name in zip(
            [("shoulder_left", "elbow_left", "wrist_left"),
             ("shoulder_right", "elbow_right", "wrist_right"),
             ("hip_left", "knee_left", "ankle_left"),
             ("hip_right", "knee_right", "ankle_right")],
            angles.keys()
        ):
            p1 = (int(keypoints[joint1].x * width), int(keypoints[joint1].y * height))
            p2 = (int(keypoints[joint2].x * width), int(keypoints[joint2].y * height))
            p3 = (int(keypoints[joint3].x * width), int(keypoints[joint3].y * height))

            # 3D effect using shadow
            shadow_offset = 3
            cv2.line(frame, (p1[0] + shadow_offset, p1[1] + shadow_offset),
                     (p2[0] + shadow_offset, p2[1] + shadow_offset), (50, 50, 50), 3)
            cv2.line(frame, (p2[0] + shadow_offset, p2[1] + shadow_offset),
                     (p3[0] + shadow_offset, p3[1] + shadow_offset), (50, 50, 50), 3)

            # Gradient-colored lines
            color = gradient_color(angles[angle_name])
            cv2.line(frame, p1, p2, color, 4)
            cv2.line(frame, p2, p3, color, 4)

            # Draw animated dotted lines
            for t in np.linspace(0, 1, 10):
                dot_x = int(p1[0] * (1 - t) + p2[0] * t)
                dot_y = int(p1[1] * (1 - t) + p2[1] * t)
                cv2.circle(frame, (dot_x, dot_y), 3, (255, 255, 255), -1)

            # Display angles
            text_position = (p2[0] + 10, p2[1] - 10)
            cv2.putText(frame, f"{angle_name}: {int(angles[angle_name])}°",
                        text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display coordinates of key points
        for joint, landmark in keypoints.items():
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.putText(frame, f"{joint}: ({x}, {y})", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Provide feedback based on angle threshold
        feedbacks = {
            "Fix Left Arm!": angles["Left Arm"] < 50 or angles["Left Arm"] > 180,
            "Fix Right Arm!": angles["Right Arm"] < 50 or angles["Right Arm"] > 180,
            "Fix Left Leg!": angles["Left Leg"] < 50 or angles["Left Leg"] > 200,
            "Fix Right Leg!": angles["Right Leg"] < 160 or angles["Right Leg"] > 200,
            "Fix Left hip!": angles["Left Arm"] < 50 or angles["Left Arm"] > 180,
            "Fix Right hip!": angles["Right Arm"] < 50 or angles["Right Arm"] > 180,
            "Fix Left shoulder!": angles["Left Leg"] < 50 or angles["Left Leg"] > 200,
            "Fix Right shoulder!": angles["Right Leg"] < 160 or angles["Right Leg"] > 200,
            "Fix Left elbow!": angles["Left Arm"] < 50 or angles["Left Arm"] > 180,
            "Fix Right elbow!": angles["Right Arm"] < 50 or angles["Right Arm"] > 180,
            "Fix Left wrist!": angles["Left Leg"] < 50 or angles["Left Leg"] > 200,
            "Fix Right wrist!": angles["Right Leg"] < 160 or angles["Right Leg"] > 200,
            "Fix Left knee!": angles["Left Arm"] < 50 or angles["Left Arm"] > 180,
            "Fix Right knee!": angles["Right Arm"] < 50 or angles["Right Arm"] > 180,
            "Fix Left ankel!": angles["Left Leg"] < 50 or angles["Left Leg"] > 200,
            "Fix Right ankel!": angles["Right Leg"] < 160 or angles["Right Leg"] > 200,

        }

        y_offset = 50
        for text, condition in feedbacks.items():
            if condition:
                cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_offset += 50

    # Show the frame with feedback
    cv2.imshow('3D Pose Detection with Angles', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
