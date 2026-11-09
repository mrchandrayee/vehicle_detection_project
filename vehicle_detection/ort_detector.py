import logging
from pathlib import Path
import onnxruntime as ort
from .preprocess import letterbox
from .decode import decode_yolo_output

log = logging.getLogger(__name__)

class OrtDetector:
    def __init__(self, model_path, confidence=0.35, names=None, device="cpu", imgsz=640):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model missing: {path}. Add your trained/exported model.")
        providers = ["CPUExecutionProvider"]
        if device.lower() in ("cuda", "gpu"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                log.warning("CUDAExecutionProvider unavailable; using CPU")
        self.session = ort.InferenceSession(str(path), providers=providers)
        inputs = self.session.get_inputs()
        if not inputs:
            raise RuntimeError("Model has no input tensor")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        self.imgsz = int(shape[2]) if len(shape)==4 and isinstance(shape[2],int) and shape[2]==shape[3] else int(imgsz)
        outputs = self.session.get_outputs()
        self.output_name = outputs[0].name if outputs else None
        self.confidence = confidence
        self.names = names
        log.info("ONNX model=%s providers=%s", path, self.session.get_providers())

    def predict(self, frame):
        tensor, scale, pad_x, pad_y = letterbox(frame, self.imgsz)
        feeds = {self.input_name: tensor}
        outputs = self.session.run([self.output_name], feeds) if self.output_name else self.session.run(None, feeds)
        return decode_yolo_output(outputs[0], scale, pad_x, pad_y, frame.shape,
                                  confidence=self.confidence, names=self.names)
