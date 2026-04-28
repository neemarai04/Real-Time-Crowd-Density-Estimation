import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import model_from_json, Model
from matplotlib import cm as c
import tempfile

# ==========================================================
#   GLOBAL CONFIG
# ==========================================================
FRAME_W = 640
FRAME_H = 480

# Thresholds for risk – tune these based on your video
DEFAULT_DENSITY_CAUTION = 20.0   # total people approx
DEFAULT_DENSITY_ALERT   = 40.0

DEFAULT_MOTION_CAUTION  = 1.0    # average motion magnitude
DEFAULT_MOTION_ALERT    = 3.0

RISK_COLORS = {
    "Safe":   (0, 255, 0),     # Green
    "Caution":(0, 255, 255),   # Yellow
    "Alert":  (0, 0, 255),     # Red
}

# ==========================================================
#   MODEL LOADING (DENSITY)
# ==========================================================
@st.cache_resource
def load_model():
    """
    Load and return the pretrained crowd-counting model.
    Cached so it doesn't reload on each rerun.
    """
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Model': Model})
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model

# ==========================================================
#   PREPROCESSING
# ==========================================================
def preprocess_frame(frame_bgr):
    """
    Preprocesses a frame for the crowd counting model.
    Assumes frame_bgr is already resized to FRAME_W x FRAME_H.
    """
    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    im = np.array(Image.fromarray(frame_rgb))

    # Normalize (0–1)
    im = im / 255.0

    # Standardization (ImageNet style)
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    # Add batch dimension
    im = np.expand_dims(im, axis=0)
    return im

# ==========================================================
#   OPTICAL FLOW
# ==========================================================
def compute_optical_flow(prev_gray, curr_gray):
    """
    Compute dense optical flow between two grayscale frames and
    return the magnitude map (movement intensity).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5,  # pyr_scale
        3,    # levels
        15,   # winsize
        3,    # iterations
        5,    # poly_n
        1.2,  # poly_sigma
        0     # flags
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag  # motion magnitude

# ==========================================================
#   GLOBAL ANALYSIS (NO ZONES)
# ==========================================================
def analyze_global(density_map, motion_mag,
                   density_caution, density_alert,
                   motion_caution, motion_alert):
    """
    Analyze the whole frame (no zones).

    density_map: 2D numpy array
    motion_mag: 2D numpy array (or None if not available)
    Returns:
        total_count (float), avg_motion (float), risk_level (str)
    """
    total_count = float(np.sum(density_map))

    if motion_mag is not None:
        avg_motion = float(np.mean(motion_mag))
    else:
        avg_motion = 0.0

    # Decide risk based on both density & motion
    risk = "Safe"

    # Alert if very high density OR very high motion
    if total_count > density_alert or avg_motion > motion_alert:
        risk = "Alert"
    # Caution for moderate density or motion
    elif total_count > density_caution or avg_motion > motion_caution:
        risk = "Caution"

    return total_count, avg_motion, risk

# ==========================================================
#   VISUALIZATION HELPERS
# ==========================================================
def draw_global_overlay(frame_bgr, risk_level, total_count, avg_motion):
    """
    Draw global info banner on the frame.
    """
    overlay = frame_bgr.copy()

    text = f"Risk: {risk_level} | Count: {int(total_count)} | Motion: {avg_motion:.2f}"
    color = RISK_COLORS[risk_level]

    # Background rectangle for text
    cv2.rectangle(
        overlay,
        (0, 0),
        (FRAME_W, 40),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        overlay,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return overlay

def make_density_heatmap(density_map):
    """
    Convert density map to a colored heatmap resized to frame size.
    """
    dm_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map) + 1e-5)
    dm_color = (c.jet(dm_norm)[:, :, :3] * 255).astype(np.uint8)
    dm_color = cv2.resize(dm_color, (FRAME_W, FRAME_H))
    return dm_color

# ==========================================================
#   MAIN STREAMLIT APP (VIDEO ONLY)
# ==========================================================
def main():
    st.title("crowd safety.ai - Global Risk Analysis")

    # Sidebar – thresholds
    st.sidebar.header("Risk Thresholds")

    density_caution = st.sidebar.number_input(
        "Density Caution Threshold (people approx)",
        min_value=1.0, max_value=10000.0,
        value=DEFAULT_DENSITY_CAUTION, step=1.0
    )
    density_alert = st.sidebar.number_input(
        "Density Alert Threshold (people approx)",
        min_value=1.0, max_value=10000.0,
        value=DEFAULT_DENSITY_ALERT, step=1.0
    )
    motion_caution = st.sidebar.number_input(
        "Motion Caution Threshold (avg magnitude)",
        min_value=0.0, max_value=50.0,
        value=DEFAULT_MOTION_CAUTION, step=0.1
    )
    motion_alert = st.sidebar.number_input(
        "Motion Alert Threshold (avg magnitude)",
        min_value=0.0, max_value=50.0,
        value=DEFAULT_MOTION_ALERT, step=0.1
    )

    frame_skip = st.sidebar.slider(
        "Process 1 frame every N frames (for speed)",
        1, 10, 2
    )

    show_heatmap = st.sidebar.checkbox("Show density heatmap", value=True)

    # Load density model
    model = load_model()

    # Video upload (for CCTV, you can replace with RTSP/Webcam)
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        heatmap_frame = st.empty() if show_heatmap else None
        stats_placeholder = st.empty()

        frame_count = 0
        prev_gray = None

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            motion_mag = None
            if prev_gray is not None:
                motion_mag = compute_optical_flow(prev_gray, gray)

            prev_gray = gray

            if frame_count % frame_skip == 0:
                # 1️⃣ Density & motion analysis
                processed = preprocess_frame(frame_resized.copy())
                density_map = model.predict(processed)
                density_map = density_map.reshape(density_map.shape[1], density_map.shape[2])

                total_count, avg_motion, risk = analyze_global(
                    density_map,
                    motion_mag=motion_mag,
                    density_caution=density_caution,
                    density_alert=density_alert,
                    motion_caution=motion_caution,
                    motion_alert=motion_alert
                )

                # 2️⃣ Draw risk overlay
                vis_frame = frame_resized.copy()
                vis_frame = draw_global_overlay(vis_frame, risk, total_count, avg_motion)

                # 3️⃣ Show main frame
                stframe.image(vis_frame, channels="BGR")

                # 4️⃣ Show density heatmap (optional)
                if show_heatmap and heatmap_frame is not None:
                    dm_heatmap = make_density_heatmap(density_map)
                    heatmap_frame.image(dm_heatmap, caption="Density Map (Heatmap)")

                # 5️⃣ Stats text
                stats_placeholder.markdown(
                    f"**Total Estimated People:** {int(total_count)}"
                    f"&nbsp;&nbsp;|&nbsp;&nbsp; **Avg Motion (Optical Flow):** {avg_motion:.2f}"
                    f"&nbsp;&nbsp;|&nbsp;&nbsp; **Risk Level:** **{risk}**",
                    unsafe_allow_html=True
                )

            frame_count += 1

        vf.release()

if __name__ == "__main__":
    main()
