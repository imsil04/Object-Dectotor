from ultralytics import YOLO 
model = YOLO("best.pt") 
model.predict("test.mp4", save=True, conf=0.4) 
