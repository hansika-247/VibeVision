import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import webbrowser


model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils
st.set_page_config(
    page_title="VibeVision - Emotion-Based Music Recommender",
    page_icon="🎶",
    layout="centered"
)

st.title("🎶 VibeVision")
st.markdown(
    """
### Your Personal Emotion-Powered Music Companion 🎧  

**VibeVision** detects your **current mood** and recommends songs that perfectly match your vibe.  
Forget endless searching—let your emotions guide the playlist!  

✨ **Features:**  
- 🎥 Real-time **emotion detection** via webcam  
- 🎭 Captures **facial & hand expressions** for accuracy  
- 🌐 Customize by **language** & **favorite singer**  
- 🎵 Instant **YouTube song recommendations**  
- 🚀 Simple, fun & interactive experience  

---
    """
)

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm,1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x-res.face_landmarks.landmark[1].x)
                lst.append(i.y-res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x-res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y-res.left_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x-res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y-res.right_hand_landmarks.landmark[8].y)
            else:
                for _ in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst, verbose=0))]

            cv2.putText(frm, f"Emotion: {pred}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(
            frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255),
                                                      thickness=-1,
                                                      circle_radius=1),
            connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm,res.left_hand_landmarks,hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm,res.right_hand_landmarks,hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm,format="bgr24")
st.subheader("🎤 Personalize Your Music")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        lang = st.text_input("🌐 Language", placeholder="e.g., Hindi, English, Telugu")
    with col2:
        singer = st.text_input("🎙️ Singer", placeholder="e.g., Arijit Singh, Taylor Swift")

st.divider()
if lang and singer and st.session_state["run"] != "false":
    st.info("🎥 Please look at the camera while we capture your emotion...")
    webrtc_streamer(
        key="key",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
                # Optionally add TURN server entry here for reliability
            ]
        }
    )
btn = st.button("🎧 Recommend Me Songs", use_container_width=True)

if btn:
    if not emotion:
        st.warning("⚠️ Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        st.success(f"✅ Emotion detected: **{emotion}**. Opening YouTube...")
        webbrowser.open(
            f"https://www.youtube.com/results?search_query={lang}+{emotion}+songs+of+{singer}"
        )
        np.save("emotion.npy",np.array([""]))
        st.session_state["run"] = "false"
