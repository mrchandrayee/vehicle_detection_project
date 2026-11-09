import numpy as np

def iou_xyxy(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-7)

class IoUTracker:
    """Small greedy IoU tracker assigning stable IDs between adjacent frames."""
    def __init__(self, match_iou=0.25, max_age=20):
        self.match_iou=match_iou; self.max_age=max_age
        self.tracks={}; self.next_id=1

    def update(self, detections):
        for item in self.tracks.values(): item["age"] += 1
        unmatched=set(self.tracks.keys())
        for det in sorted(detections, key=lambda d:d.confidence, reverse=True):
            best_id=None; best=0.0
            box=(det.x1,det.y1,det.x2,det.y2)
            for tid in list(unmatched):
                tr=self.tracks[tid]
                if tr["class_id"] != det.class_id: continue
                score=iou_xyxy(box,tr["box"])
                if score>best: best=score; best_id=tid
            if best_id is not None and best>=self.match_iou:
                tid=best_id; unmatched.remove(tid)
                self.tracks[tid]={"box":box,"class_id":det.class_id,"age":0}
            else:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]={"box":box,"class_id":det.class_id,"age":0}
            det.track_id=tid
        self.tracks={k:v for k,v in self.tracks.items() if v["age"]<=self.max_age}
        return detections
