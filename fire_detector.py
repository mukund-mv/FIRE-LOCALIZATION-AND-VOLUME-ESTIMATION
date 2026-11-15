"""YOLO fire detection (your existing Ultralytics logic, wrapped in a class)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    cx: float
    cy: float
    width: int
    height: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class FireDetector:
    def __init__(self, model_path, conf_threshold: float = 0.5,
                 fire_classes: Optional[Sequence[str]] = None, device: Optional[str] = None):
        model_path = str(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}\n"
                "Check YOLO_MODEL_PATH in config.py.")
        self.model = YOLO(model_path, task="detect")
        self.labels = self.model.names
        self.conf_threshold = float(conf_threshold)
        self.fire_classes = {c.lower() for c in fire_classes} if fire_classes else None
        self.device = device

    def warmup(self, shape=(480, 640, 3)) -> None:
        self.detect(np.zeros(shape, dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO on one BGR frame. Returns detections sorted by confidence (best first)."""
        kwargs = {"verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model(frame, **kwargs)
        detections = results[0].boxes

        h, w = frame.shape[:2]
        out: List[Detection] = []
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = self.labels[classidx]
            conf = detections[i].conf.item()

            if conf <= self.conf_threshold:
                continue
            if self.fire_classes is not None and str(classname).lower() not in self.fire_classes:
                continue

            xmin, xmax = max(0, int(xmin)), min(w - 1, int(xmax))
            ymin, ymax = max(0, int(ymin)), min(h - 1, int(ymax))
            if xmax <= xmin or ymax <= ymin:
                continue

            out.append(Detection(
                x1=xmin, y1=ymin, x2=xmax, y2=ymax,
                cx=(xmin + xmax) / 2.0, cy=(ymin + ymax) / 2.0,
                width=xmax - xmin, height=ymax - ymin,
                confidence=float(conf), class_id=classidx, class_name=str(classname)))
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out