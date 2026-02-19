"""
Detection utilities for DeepSORT
"""
import numpy as np
import cv2


class Detection:
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def xywh_to_xyxy(bbox_xywh):
    """
    Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2)
    """
    x, y, w, h = bbox_xywh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2


def xywh_to_tlwh(bbox_xywh):
    """
    Convert bounding box from (center_x, center_y, w, h) to (top_left_x, top_left_y, w, h)
    """
    x, y, w, h = bbox_xywh
    t = x - w / 2
    l = y - h / 2
    return t, l, w, h


def tlwh_to_xyxy(bbox_tlwh):
    """
    Convert bounding box from (top_left_x, top_left_y, w, h) to (x1, y1, x2, y2)
    """
    x, y, w, h = bbox_tlwh
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return x1, y1, x2, y2


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
