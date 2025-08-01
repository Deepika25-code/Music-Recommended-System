import streamlit as st
from streamlit_webrtc import webrtc_streamer
import base64
import av
import cv2 
import numpy as np 
import mediapipe as mp 
import tensorflow as tf
from keras.models import load_model
import webbrowser
import tensorflow as tf


try:
    model = tf.keras.models.load_model("model.h5") 
    
except Exception as e:
    print("Error loading model:", e)

label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Smart Music Recommendation System")
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


img_base64 = get_base64_image("background.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    div.stButton > button {{
        background-color: #545454;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
    }}

    div.stButton > button:hover {{
        background-color: #0056b3;
    }}

    .stTextInput>div>div>input {{
        background-color: #f0f8ff;
        color: #000;
        font-weight: bold;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"

class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("Singer")

if lang and singer and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)

btn = st.button("Recommend Songs")


if btn:
    if not(emotion):
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://open.spotify.com/search/{singer}%20{lang}%20{emotion}%20hits")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
