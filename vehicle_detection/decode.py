import numpy as np
from .types import Detection

def nms(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    area = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= threshold]
    return keep

def decode_yolo_output(output, scale, pad_x, pad_y, frame_shape,
                       confidence=0.35, iou=0.45, names=None):
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {np.asarray(output).shape}")

    if arr.shape[-1] == 6:  # common end-to-end xyxy, confidence, class
        boxes = arr[:, :4].astype(float)
        scores = arr[:, 4].astype(float)
        class_ids = arr[:, 5].astype(int)
        if boxes.size and np.nanmax(np.abs(boxes)) <= 1.5:
            boxes[:, [0, 2]] *= 640
            boxes[:, [1, 3]] *= 640
    else:
        # YOLOv8 commonly exports [4+C,N]; transpose to [N,4+C].
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 256:
            arr = arr.T
        if arr.shape[1] < 5:
            raise ValueError(f"Cannot interpret output shape {arr.shape}")
        # For YOLOv5, column 4 is objectness; YOLOv8 has class scores from col 4.
        cls = arr[:, 5:] if arr.shape[1] >= 6 else arr[:, 4:]
        if arr.shape[1] >= 6 and arr[:, 4].min() >= 0 and arr[:, 4].max() <= 1:
            scores_matrix = cls * arr[:, 4:5]
        else:
            scores_matrix = arr[:, 4:]
        class_ids = scores_matrix.argmax(axis=1)
        scores = scores_matrix[np.arange(len(scores_matrix)), class_ids]
        xywh = arr[:, :4].astype(float)
        boxes = np.column_stack((xywh[:,0]-xywh[:,2]/2, xywh[:,1]-xywh[:,3]/2,
                                 xywh[:,0]+xywh[:,2]/2, xywh[:,1]+xywh[:,3]/2))

    h, w = frame_shape[:2]
    found = []
    for cid in np.unique(class_ids):
        ids = np.where((class_ids == cid) & (scores >= confidence))[0]
        if not len(ids):
            continue
        for local_i in nms(boxes[ids], scores[ids], iou):
            idx = ids[local_i]
            b = boxes[idx].copy()
            b[[0,2]] = (b[[0,2]] - pad_x) / scale
            b[[1,3]] = (b[[1,3]] - pad_y) / scale
            x1,y1,x2,y2 = b
            x1 = int(np.clip(x1, 0, w-1)); x2 = int(np.clip(x2, 0, w-1))
            y1 = int(np.clip(y1, 0, h-1)); y2 = int(np.clip(y2, 0, h-1))
            cid = int(class_ids[idx])
            label = names[cid] if names and cid < len(names) else f"class_{cid}"
            found.append(Detection(x1,y1,x2,y2,float(scores[idx]),cid,label))
    return found
