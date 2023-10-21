import tensorflow as tf
from typing import List
import cv2
import os 
import mediapipe as mp

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def detect_mouth(video_path):
    # Initialize the Mediapipe face detection model
    mp_face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    mouth_bounding_boxes = []

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (Mediapipe requires RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the face detection model
        results = mp_face_detection.process(frame_rgb)

        # Check if a face is detected in the frame
        if results.detections:
            # Get the bounding box of the face
            face_bounding_box = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = frame.shape

            # Calculate the coordinates of the mouth bounding box
            mouth_left_x = int((face_bounding_box.xmin + 0.1 * face_bounding_box.width) * iw)
            mouth_right_x = int((face_bounding_box.xmin + 0.9 * face_bounding_box.width) * iw)
            mouth_top_y = int((face_bounding_box.ymin + 0.6 * face_bounding_box.height) * ih)
            mouth_bottom_y = int((face_bounding_box.ymin + face_bounding_box.height) * ih)

            mouth_bounding_boxes.append((mouth_left_x, mouth_top_y, mouth_right_x, mouth_bottom_y))

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return mouth_bounding_boxes


# def load_video(path: str) -> tf.Tensor:
#     # Detect mouth bounding box coordinates
#     mouth_bounding_boxes = detect_mouth(path)

#     desired_height = 46  # Replace with your desired height
#     desired_width = 140  # Replace with your desired width

#     # Open the video file
#     cap = cv2.VideoCapture(path)

#     frames = []
#     frame_count = 0
#     for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Crop and resize the frame using the mouth bounding box coordinates
#         for mouth_box in mouth_bounding_boxes:
#             mouth_left_x, mouth_top_y, mouth_right_x, mouth_bottom_y = mouth_box
#             frame_cropped_resized = frame[mouth_top_y:mouth_bottom_y, mouth_left_x:mouth_right_x, :]
#             frame_cropped_resized = tf.image.rgb_to_grayscale(frame_cropped_resized)
#             frame_cropped_resized = tf.image.resize(frame_cropped_resized, (desired_height, desired_width))
#             frame_cropped_resized = tf.reshape(frame_cropped_resized, (desired_height, desired_width, 1))
        

#             frames.append(frame_cropped_resized)

#     cap.release()

#     mean = tf.math.reduce_mean(frames)
#     std = tf.math.reduce_std(tf.cast(frames, tf.float32))
#     processed_frames = tf.cast((frames - mean), tf.float32) / std

#     # Convert processed_frames back to uint8
#     # processed_frames = tf.clip_by_value(processed_frames, -1, 1)
#     # processed_frames = tf.cast(((processed_frames + 1) * 127.5), tf.uint8)

#     if len(processed_frames) < 75:
#         # Pad preprocessed_frames with zeros to make its length equal to 75
#         processed_frames = tf.pad(processed_frames, [[0, 75 - len(processed_frames)], [0, 0], [0, 0], [0, 0]])
#     elif len(processed_frames) > 75:
#         # Truncate preprocessed_frames to make its length equal to 75
#         processed_frames = processed_frames[:75]
#     else:
#         pad_width = 75 - len(processed_frames)
#         processed_frames = tf.pad(processed_frames, [[0, pad_width], [0, 0], [0, 0], [0, 0]])

    

#     return processed_frames

def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    with tf.device('/cpu:0'):
        frames = load_video(video_path) 
    # alignments = load_alignments(alignment_path)
    
    return frames
