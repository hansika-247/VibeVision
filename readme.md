# 🎵 VibeVision – Mood Based Music Recommendation 🎶  

VibeVision is an AI-powered web app that detects your **facial emotions in real-time** and recommends music that matches your mood.  
It combines **Computer Vision, Deep Learning, and Streamlit** to deliver a fun and interactive experience.  

---

## ✨ Features  
- 🎥 **Real-time face detection** using your webcam  
- 😀 **Emotion recognition** (Happy, Sad, Angry, Surprised, Neutral, etc.)  
- 🎶 **Smart music recommendation** based on your detected mood  
- 🌸 **Pastel themed UI** for a soothing user experience  
- ⚡ Built with **Streamlit** for simplicity and interactivity  

---

## ⚡ Quick Start  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/your-username/mood-music-app.git
cd mood-music-app
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
### 3️⃣ Run the app
```bash
streamlit run musicapp.py

--- 

## 🚀 Tech Stack  
- **Frontend & UI** → [Streamlit](https://streamlit.io/) for interactive, real-time web interface  
- **Machine Learning** → [TensorFlow/Keras](https://www.tensorflow.org/) for training & inference (`model.h5`)  
- **Computer Vision** → [MediaPipe](https://mediapipe.dev/) for face & hand landmark detection  
- **Image Processing** → [OpenCV](https://opencv.org/) for frame capture & preprocessing  
- **Data Handling** → [NumPy](https://numpy.org/) for efficient dataset storage  
- **Deployment** → Streamlit Cloud / Hugging Face Spaces  

---
## 🔮 Future Improvements

🎼 Spotify / YouTube API integration → Directly play songs instead of opening search results
🧠 More emotions & fine-grained detection → Expand beyond 6–7 emotions for richer personalization
📊 User emotion history dashboard → Track mood trends with graphs & insights
🎨 Advanced UI customization → Dark mode, themes, animations
