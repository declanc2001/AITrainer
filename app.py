import mediapipe as mp
import streamlit as st
import inspect

st.write("mediapipe version:", getattr(mp, "__version__", "no __version__"))
st.write("mediapipe module path:", getattr(mp, "__file__", "no __file__"))
st.write("has solutions:", hasattr(mp, "solutions"))

# hard fail with clear message
if not hasattr(mp, "solutions"):
    raise RuntimeError("mediapipe imported but mp.solutions is missing. Likely module shadowing or broken install.")

# ---------- Core math ----------
def angle_between(v1, v2):
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan
    cosang = np.dot(v1, v2) / denom
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def extract_angle_series(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    angles, times = [], []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w = frame.shape[:2]

            def pt(idx):
                p = lm[idx]
                return np.array([p.x * w, p.y * h], dtype=np.float32)

            ls = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
            rs = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
            lh = pt(mp_pose.PoseLandmark.LEFT_HIP)
            rh = pt(mp_pose.PoseLandmark.RIGHT_HIP)

            ang = angle_between(ls - rs, lh - rh)
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

files = st.file_uploader("Upload 4 swing videos (.mp4)", type=["mp4", "mov", "m4v"], accept_multiple_files=True)

if files and len(files) != 4:
    st.warning(f"Please upload exactly 4 videos (you uploaded {len(files)}).")

if st.button("Compute repeatability", type="primary", disabled=(not files or len(files) != 4)):
    with st.spinner("Analyzing swings..."):
        score, df = score_golfer(files)

    st.metric("Repeatability (0–1)", f"{score:.3f}")
    st.subheader("Extracted features (per swing)")
    st.dataframe(df, use_container_width=True)

    st.caption("Note: This is a 2D pose-based proxy metric. Results depend on camera angle and visibility.")
