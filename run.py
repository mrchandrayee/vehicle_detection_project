import argparse
from pathlib import Path
import cv2
from vehicle_detection.ort_detector import OrtDetector
from vehicle_detection.tracker import IoUTracker

def read_names(path):
    p=Path(path)
    return [s.strip() for s in p.read_text().splitlines() if s.strip()] if p.exists() else None

def main():
    ap=argparse.ArgumentParser(description="YOLO ONNX vehicle detection and tracking")
    ap.add_argument("--source",default="0",help="Video/image path or webcam index")
    ap.add_argument("--model",default="models/vehicle.onnx")
    ap.add_argument("--names",default="classes.txt")
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--conf",type=float,default=0.35)
    ap.add_argument("--output",default="runs/vehicle_output.mp4")
    args=ap.parse_args()
    names=read_names(args.names)
    detector=OrtDetector(args.model,args.conf,names,args.device,args.imgsz)
    tracker=IoUTracker()
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open input: {args.source}")
    is_camera=isinstance(source,int)
    writer=None
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    palette=[(255,80,255),(0,220,255),(255,180,0),(0,255,80),(255,0,100),(150,80,255)]
    while True:
        ok,frame=cap.read()
        if not ok: break
        detections=tracker.update(detector.predict(frame))
        counts={}
        for det in detections:
            counts[det.class_name]=counts.get(det.class_name,0)+1
            color=palette[det.track_id%len(palette)]
            cv2.rectangle(frame,(det.x1,det.y1),(det.x2,det.y2),color,2)
            label=f"ID {det.track_id} | {det.class_name} {det.confidence:.2f}"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.48,1)
            y=max(0,det.y1-th-7)
            cv2.rectangle(frame,(det.x1,y),(det.x1+tw+6,y+th+7),color,-1)
            cv2.putText(frame,label,(det.x1+3,y+th+2),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),1,cv2.LINE_AA)
        summary="  ".join(f"{k}: {v}" for k,v in sorted(counts.items()))
        cv2.putText(frame,summary or "No detections",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
        if not is_camera and writer is None:
            fps=cap.get(cv2.CAP_PROP_FPS) or 25
            h,w=frame.shape[:2]
            writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
        if writer is not None: writer.write(frame)
        cv2.imshow("Vehicle Detection - press q to quit",frame)
        if cv2.waitKey(1)&0xFF==ord("q"): break
    cap.release()
    if writer is not None:
        writer.release(); print("Saved:",args.output)
    cv2.destroyAllWindows()

if __name__=="__main__": main()
