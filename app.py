import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

def download_model(path="pose_landmarker.task"):
    # Download once per container
    try:
        with open(path, "rb"):
            return path
    except FileNotFoundError:
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return path

@st.cache_resource
def make_landmarker():
    model_path = download_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1
    )
    return vision.PoseLandmarker.create_from_options(options)

def extract_angle_series(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    landmarker = make_landmarker()

    angles = []
    times = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm = result.pose_landmarks[0]  # first pose

            def pt(i):
                return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

            # MediaPipe landmark indices (Pose):
            # 11 L_SHOULDER, 12 R_SHOULDER, 23 L_HIP, 24 R_HIP
            ls = pt(11); rs = pt(12); lh = pt(23); rh = pt(24)

            shoulder_vec = ls - rs
            hip_vec = lh - rh

            denom = (np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec))
            if denom == 0:
                ang = np.nan
            else:
                cosang = np.dot(shoulder_vec, hip_vec) / denom
                ang = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
        else:
            ang = np.nan

        angles.append(ang)
        times.append(frame_idx / fps)
        frame_idx += 1

    cap.release()

    times = np.array(times, dtype=float)
    angles = pd.Series(angles).interpolate(limit_direction="both").to_numpy(dtype=float)
    return times, angles


def features_from_series(times, angles):
    # ignore first 0.25 sec
    mask = times >= 0.25
    if mask.sum() < 5:
        mask = np.ones_like(times, dtype=bool)

    t = times[mask]
    a = angles[mask]

    # smooth
    a = pd.Series(a).rolling(window=7, center=True).median().interpolate(limit_direction="both").to_numpy()

    peak = float(np.percentile(a, 95))
    peak_idx = int(np.argmax(a))
    t_to_peak_frac = float(peak_idx / max(1, len(a) - 1))
    mean_angle = float(np.mean(a))

    return {"peak_deg": peak, "t_to_peak_frac": t_to_peak_frac, "mean_deg": mean_angle}

def repeatability_score_0_1(features_list):
    df = pd.DataFrame(features_list)

    std_peak = float(df["peak_deg"].std(ddof=0))
    std_ttp  = float(df["t_to_peak_frac"].std(ddof=0))
    std_mean = float(df["mean_deg"].std(ddof=0))

    peak_range = 25.0
    ttp_range  = 0.35
    mean_range = 15.0

    p_peak = min(std_peak / peak_range, 1.0)
    p_ttp  = min(std_ttp  / ttp_range,  1.0)
    p_mean = min(std_mean / mean_range, 1.0)

    penalty = 0.45*p_peak + 0.35*p_ttp + 0.20*p_mean
    score = max(0.0, min(1.0, 1.0 - penalty))
    return score, df

def save_uploaded_file(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name

def score_golfer(uploaded_files):
    feats = []
    for uf in uploaded_files:
        path = save_uploaded_file(uf)
        times, angles = extract_angle_series(path)
        feats.append(features_from_series(times, angles))
    score, df = repeatability_score_0_1(feats)
    return score, df

# ---------- UI ----------
st.set_page_config(page_title="Golf Swing Repeatability", layout="centered")
st.title("🏌️ Golf Swing Repeatability Scorer")
st.write("Upload **4 swing videos** for a golfer and press **Compute** to get a repeatability score (0–1).")
st.write("I am Declan Carr-Mcelhatton, a final year student at Portsmouth University and this is my golf swing analysis tool, mediapipe pose selection is used to measure how consistent a golfers movement is, video submitted must be consistent with camera angle and with no external  visual noise or movement otherwise results will vary wildly.")

if "score" not in st.session_state:
    st.session_state.score = None
if "df" not in st.session_state:
    st.session_state.df = None


files = st.file_uploader("Upload 4 swing videos (.mp4)", type=["mp4", "mov", "m4v"], accept_multiple_files=True)

if files and len(files) != 4:
    st.warning(f"Please upload exactly 4 videos (you uploaded {len(files)}).")

if st.button("Compute repeatability", type="primary", disabled=(not files or len(files) != 4)):
    with st.spinner("Analyzing swings..."):
        score, df = score_golfer(files)
        st.session_state.score = score
        st.session_state.df = df


if st.session_state.score is not None:
    score = st.session_state.score
    df = st.session_state.df

    st.metric("Repeatability (0–1)", f"{score:.3f}")

    st.subheader("What does this score mean?")
    if score < 0.3:
        st.error("❌ Swing needs work — high variation between swings. Time to hit the driving range.")
    elif score < 0.6:
        st.warning("⚠️ Not bad — some consistency, but there is still room for improvement.")
    else:
        st.success("✅ Very good — strong repeatability and reliable swing mechanics.")

    st.subheader("Extracted features (per swing)")
    st.dataframe(df, use_container_width=True)

    st.caption("Note: This is a 2D pose-based proxy metric. Results depend on camera angle and visibility.")
else:
    st.info("Upload 4 videos and click **Compute repeatability** to see your score.")
