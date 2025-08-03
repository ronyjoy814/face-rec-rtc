# coding: utf-8

"""
Face Recognition Package

A modular face recognition system using Azure AI Vision Face API.

Main Components:
- FaceGroupManager: Main public API for group coordination and frame processing
- KnownFaceGroup: Known face management
- UnknownFaceGroup: Unknown face management
- BaseGroupManager: Common Azure Face API operations

Features:
- Dual-group architecture (known + unknown groups)
- Auto-training with confidence thresholds
- Timestamp tracking for face additions
- Professional unknown person management
- Modular, maintainable design
"""

from .face_group_manager import FaceGroupManager, FrameProcessingResult, DetectedFace
from .known_face_group import KnownFaceGroup
from .unknown_face_group import UnknownFaceGroup
from .base_group_manager import BaseGroupManager

__version__ = "1.0.0"
__author__ = "FastRTC Team"

__all__ = [
	"FaceGroupManager",
	"FrameProcessingResult",
	"DetectedFace",
	"KnownFaceGroup",
	"UnknownFaceGroup",
	"BaseGroupManager"
]
