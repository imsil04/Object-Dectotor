import streamlit as st
import cv2, tempfile, subprocess, imageio_ffmpeg
from ultralytics import YOLO

# Agar naya rickshaw model use karna hai to yahan naam badlo (jaise "rickshaw2.pt")
MODEL1 = "best.pt"
MODEL2 = "rickshaw.pt"

st.set_page_config(page_title="Object Detector")


@st.cache_resource
def load_models():
    return YOLO(MODEL1), YOLO(MODEL2)


model, model2 = load_models()

st.title("Video Object Detection")
st.write("Video upload karo (chhoti video, 10-15 second behtar hai)")

f = st.file_uploader("Video chuno", type=["mp4", "avi", "mov"])

if f and st.button("Detect"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(f.read())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    raw = tmp.name + "_raw.mp4"
    out = tmp.name + "_out.mp4"
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    bar = st.progress(0, text="Processing...")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, conf=0.4, verbose=False)[0]
        r2 = model2(frame, conf=0.25, verbose=False)[0]
        img = r.plot()
        img = r2.plot(img=img)
        writer.write(img)
        i += 1
        bar.progress(min(i / total, 1.0), text=f"Frame {i}/{total}")
    cap.release()
    writer.release()

    # browser ke liye H.264 mein convert
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw,
         "-vcodec", "libx264", "-pix_fmt", "yuv420p", out],
        capture_output=True,
    )

    bar.empty()
    st.success("Ho gaya!")
    st.video(out)
    with open(out, "rb") as v:
        st.download_button("Download result", v, file_name="result.mp4")
