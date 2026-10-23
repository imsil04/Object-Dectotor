from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2, shutil, uuid, os, subprocess, imageio_ffmpeg

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = YOLO("best.pt")
model2 = YOLO("rickshaw.pt")
os.makedirs("outputs", exist_ok=True)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    inp = f"in_{uid}.mp4"
    raw = f"outputs/raw_{uid}.mp4"
    out = f"outputs/{uid}.mp4"

    with open(inp, "wb") as f:                    # upload ki hui video save
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(inp)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:                          # har frame par detect
        ok, frame = cap.read()
        if not ok:
            break
        r = model(frame, conf=0.4, verbose=False)[0]
        r2 = model2(frame, conf=0.5, verbose=False)[0]
        writer.write(r2.plot(img=r.plot()))
    cap.release(); writer.release()

    # browser ke liye H.264 mein convert
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p", out],
                   capture_output=True)
    os.remove(inp)
    return FileResponse(out, media_type="video/mp4")