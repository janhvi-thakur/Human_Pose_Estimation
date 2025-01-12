import cv2
import mediapipe as mp

#initialize mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils #for drawing landmarks

# capture video: replace with 'path-to-video-.mp4' for a file or 0 for webcam
video_source = 0 #0 for live webcam, or replace with video file path
cap = cv2.VideoCapture(video_source)
#cap = cv2.VideoCapture(r'C:\Users\ADMIN\Downloads\run_video.mp4')

#set video resolution for webcam (optional, adjust as needed for your laptop)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

#initializes mediapipe pose
with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video capture ended.")
            break
        # Flip frame horizontally for a mirror-like effect (optional)
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connections
            )

        # Resize frame to fit the screen (optional)
        display_frame = cv2.resize(frame, (960, 540))  # Adjust width and height as per your screen size

        # Display the processed frame
        cv2.imshow('Pose Estimation', display_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
