import streamlit as st
import os
import torch
from transformers import pipeline
from accelerate.test_utils.testing import get_backend
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
DEVICE, _, _ = get_backend()
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE, pool=True)

st.title("Camera Feed")
frame_placeholder = st.empty()
### Load reference image embeddings

print(os.listdir("./reference_images"))
ref_embeddings = [(img_path, pipe(f"reference_images/{img_path}")[0]) for img_path in os.listdir("./reference_images")]

import cv2
from torch.nn.functional import cosine_similarity
ip_address = ''  # Replace with the IP address of your camera
port = ''                # Replace with the port number for your camera
username = ''          # Replace with the username for your camera
password = ''
url_640x480 = f"rtsp://{username}:{password}@{ip_address}:{port}/stream2"
url_1080p = f"rtsp://{username}:{password}@{ip_address}:{port}/stream1"
rtsp_url = url_640x480  # Set it to either `url_640x480` or `url_1080p` based on the desired resolution
#cap = cv2.VideoCapture(rtsp_url)
cap = cv2.VideoCapture(0)
# Video writer to encode frames into H.264 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Open the RTSP stream


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Check if the RTSP stream is opened successfully
try:
    if not cap.isOpened():
        print("Failed to open RTSP stream")

    while True:
        ret, frame = cap.read()  
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect faces
        counter = 0
        if len(faces) > 0:
            
            for (x, y, w, h) in faces:
                cv2.imwrite(f"./data/tmp_img{counter}.png", frame[max(0,x-100):x+w+100, max(0, y-100):y+h+100])
                embedding = pipe(f"./data/tmp_img{counter}.png")
                similarity_score = [(ref_emb, cosine_similarity(torch.Tensor(embedding), torch.Tensor(ref_emb[1]), dim=1)) for ref_emb in ref_embeddings]
                matching_details = max(similarity_score, key=lambda x: x[1])
                #print("person: ", max(similarity_score, key=lambda x: x[1]))
                counter += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
                cv2.putText(frame, matching_details[0][0].split(".")[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        #cv2.imshow('Face Detection', frame)
        frame_placeholder.image(frame)
        
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            os.unlink("./tmp/tmp_video.mov")
            break

    # Release the RTSP stream and close the window
except Exception as e:
    print(e)
finally:
    cap.release()
    cv2.destroyAllWindows()

