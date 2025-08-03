# coding: utf-8

"""
Face Recognition Package

A modular face recognition system using Azure AI Vision Face API with quad-group architecture.

Main Components:
- FaceRecognitionManager: Main public API
- FaceGroupManager: Group coordination and frame processing
- KnownFaceGroup: Known face management with dual primary/secondary groups
- UnknownFaceGroup: Unknown face management with dual primary/secondary groups
- BaseGroupManager: Common Azure Face API operations

Features:
- Quad-group architecture (2 known + 2 unknown groups)
- Dual async identification calls per frame
- Auto-training with confidence thresholds
- Timestamp tracking for face additions
- Professional unknown person management
- Modular, maintainable design
"""

from .face_recognition_manager import FaceRecognitionManager
from .face_group_manager import FaceGroupManager, FrameProcessingResult, DetectedFace
from .known_face_group import KnownFaceGroup
from .unknown_face_group import UnknownFaceGroup
from .base_group_manager import BaseGroupManager

__version__ = "1.0.0"
__author__ = "FastRTC Team"

__all__ = [
	"FaceRecognitionManager",
	"FaceGroupManager", 
	"FrameProcessingResult",
	"DetectedFace",
	"KnownFaceGroup",
	"UnknownFaceGroup", 
	"BaseGroupManager"
]
