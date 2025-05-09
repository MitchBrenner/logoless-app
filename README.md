# 🔥 LogoLess

**Remove TikTok watermarks with ease.**  
LogoLess is a full-stack mobile app that allows users to upload TikTok videos and automatically blur out the watermark using computer vision. The app detects the watermark region with OpenCV and applies a Gaussian blur, giving you a clean, share-ready video.

---

## ✨ Features

- 📦 Upload `.mp4` TikTok videos directly from your phone
- 🎯 Automatically detects watermark location using OpenCV
- 🔄 Moves the blur intelligently during playback
- 📱 Plays the processed video immediately in-app
- 💾 Allows saving the clean video to your camera roll

---

### 🎬 Demo

![LogoLess demo](./demo.gif)

## 🧠 Tech Stack

### ⚙️ Backend (Python + FastAPI)
- **FastAPI** for the API server
- **OpenCV** to detect and blur watermarks with template detection

### 📱 Frontend (React Native + Expo)
- **expo-video** to play processed videos
- **expo-document-picker** to upload `.mp4` videos
- **expo-file-system** to handle blob > file conversions
- **expo-media-library** to save videos to the device
- UI built with `React Native` and `Hooks`

---

## 🚀 Setup Instructions

### Clone the repository to your device:
```bash
git clone https://github.com/mitchbrenner/logoless.git
cd logoless
```

### 🐍 Backend (FastAPI + OpenCV)

#### 1. Set up a Python environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### 2. Install dependencies
```bash
pip install fastapi uvicorn opencv-python python-multipart
```

#### 3. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Make sure to update app.main to your actual Python file path if needed.
- Ensure the server runs on the same Wi-Fi network as your mobile device.

### 📱 Frontend (Expo App)

#### 1. Install dependencies:
```bash
npm install
npx expo install expo-document-picker expo-video expo-file-system expo-media-library
```

#### 🔁 2. Configure API URL for Mobile Devices

To ensure your Expo app can talk to the FastAPI server from your phone, you **must not use `localhost`**. Your phone cannot access `localhost` on your computer — it needs your **computer’s local IP address**.

##### ✅ Option 1: Use an environment variable (recommended)
Create a `.env` file in the root of your project:
   ```env
   EXPO_PUBLIC_IP_ADDRESS=http://192.168.X.X:8000
   ```

##### Option 2: Replace route in `index.tsx`

#### 3. Run the app:
```bash
npx expo start
```


### 🧠 Tech Stack

| Tech             | Why It Was Chosen                                               |
|------------------|------------------------------------------------------------------|
| **FastAPI**       | Fast, async Python API framework with easy routing              |
| **OpenCV**        | Powerful, mature library for image & video processing           |
| **Expo (React Native)** | Rapid mobile development with access to native APIs |


### 🧠 What is Template Detection?

Template detection is a computer vision technique used to locate a specific image pattern (the **template**) within a larger image or video frame. In LogoLess, the template is a cropped version of the TikTok watermark.

Using OpenCV’s `matchTemplate` function, the app compares the template against a specific region in each frame of the video. This generates a **match confidence score** — the higher the score, the more likely the watermark is present in that location.

To reduce false positives caused by noise or similar-looking shapes (known as **spikes** or random high scores), the algorithm:
- Scans multiple consecutive frames
- Requires a **minimum number of strong matches (hits)** before confirming the watermark
- Only applies the blur once that confidence threshold is met

This ensures the blur is only applied if the watermark is **consistently detected** over time — increasing accuracy and avoiding unnecessary modifications.

### 📁 Why I Used File Picker Instead of Camera Roll

To ensure consistent `.mp4` support and reliable access to video files, I chose to use `expo-document-picker`. This opens the device's **file directory** instead of the photo gallery, allowing users to explicitly select `.mp4` files. 
The result is a smoother, more predictable video upload experience for the user.

### 📸 How to Use the App

1.	Start the FastAPI server (uvicorn app.main:app --reload --host 0.0.0.0 --port 8000)
2.	Run the Expo app (npx expo start)
3.	Scan the QR code using Expo Go, or open it in a simulator
4.	Tap “Upload TikTok Video” to choose an .mp4 file
5.	Wait for the processing (a loading spinner will appear)
6.	Preview the result with native controls
7.	Tap “Save Video” to save it to your camera roll or clear to start over

### 🧪 Development Notes
- The backend includes logic to move the blur region after 5 seconds, mimicking the tik tok watermark animation
- If no watermark is detected, the server responds with { "success": false } — and the app alerts the user

### Improvements
- Document code

