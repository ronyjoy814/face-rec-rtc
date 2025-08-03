# coding: utf-8

"""
Face Group Manager

Coordinates and manages both known and unknown face groups for comprehensive
face recognition operations. Provides unified interface for face detection,
identification, training, and person management across multiple groups.

Key Features:
    - Unified face group coordination
    - Cross-group face identification
    - Centralized training management
    - Person management across groups
    - Statistics and reporting
"""

import logging
import time
import queue
from datetime import datetime
import threading
import numpy as np
import os
import json
import cv2
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from azure.ai.vision.face.models import HeadPose, BlurLevel, BlurProperties, ExposureLevel, ExposureProperties

MAX_HEAD_POSE_ANGLE = 30

from .known_face_group import KnownFaceGroup
from .unknown_face_group import UnknownFaceGroup


@dataclass
class DetectedFace:
    """Represents a detected face with identification results"""
    face_id: str
    person_id: Optional[str]
    person_name: str
    confidence: float
    is_known: bool
    bounding_box: Dict[str, int]
    face_crop: Optional[np.ndarray] = None


@dataclass
class FrameProcessingResult:
    """Result of processing a frame for face recognition"""
    detected_faces: List[DetectedFace]
    total_faces: int
    known_faces: int
    unknown_faces: int
    processing_time: float
    annotated_frame: np.ndarray


class FaceGroupManager:
    """
    Face Group Manager

    Coordinates and manages both known and unknown face groups to provide
    comprehensive face recognition capabilities. Handles face detection,
    identification, training, and person management across multiple groups.
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        group_id: str,
        group_name: str,
        max_faces_per_person: int = 10,
        face_similarity_threshold: float = 0.8,
        auto_train_unknown: bool = True,
        unknown_confidence_threshold: float = 0.8,
        save_known_images: bool = True,
        save_unknown_images: bool = True,
        save_processed_frames: bool = True,
        cleanup_unknown_images: bool = False,
        cleanup_known_group: bool = False,
        base_save_directory: str = "face_recognition/saved_images",
        logger: Optional[logging.Logger] = None
    ):
        """Initialize face group manager"""
        
        self.endpoint = endpoint
        self.api_key = api_key
        self.group_id = group_id
        self.group_name = group_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize known face group
        self.known_group = KnownFaceGroup(
            endpoint=endpoint,
            api_key=api_key,
            group_id=group_id,
            group_name=group_name,
            max_faces_per_person=max_faces_per_person,
            face_similarity_threshold=face_similarity_threshold,
            save_known_images=save_known_images,
            cleanup_known_group=cleanup_known_group,
            logger=logger
        )
        
        # Initialize unknown face group
        self.unknown_group = UnknownFaceGroup(
            endpoint=endpoint,
            api_key=api_key,
            group_id=group_id,
            group_name=group_name,
            max_faces_per_person=max_faces_per_person,
            face_similarity_threshold=face_similarity_threshold,
            auto_train_unknown=auto_train_unknown,
            unknown_confidence_threshold=unknown_confidence_threshold,
            save_unknown_images=save_unknown_images,
            cleanup_unknown_images=cleanup_unknown_images,
            logger=logger
        )

        # Queue system for background processing
        self.processing_queue = queue.Queue()
        self.queue_processor_thread = None
        self.queue_processing_active = False

        # File system configuration
        self.save_processed_frames = save_processed_frames
        self.base_save_directory = Path(base_save_directory)
        self.file_lock = threading.Lock()  # Thread-safe file operations

        # Initialize directory structure
        self._initialize_directory_structure()

        # Set group manager reference for file operations
        self.known_group.set_group_manager(self)
        self.unknown_group.set_group_manager(self)

        self.logger.info(f"Initialized FaceGroupManager for: {group_name}")
        self.logger.info(f"File system base directory: {self.base_save_directory}")

    def process_frame(self, frame: np.ndarray, auto_add_faces: bool = True,
                               similarity_threshold: float = 0.8, confidence_threshold: float = 0.8,
                               time_gap_seconds: int = 0, max_faces_per_person: int = 5,
                               quality_check: bool = True) -> FrameProcessingResult:
        """
        Process frame with immediate response and background processing

        IMMEDIATE PROCESSING (Fast Response):
        1. Detect all faces in frame
        2. Identify faces against known and unknown groups (read-only)
        3. Apply tags/labels to each detected face
        4. Create annotated frame with bounding boxes and labels
        5. Return FrameProcessingResult immediately

        BACKGROUND PROCESSING (Queued):
        6. Add new faces to persons (time-consuming)
        7. Create new unknown persons
        8. Quality checks and similarity validation
        9. Training coordination
        10. File system operations (saving face crops, updating JSON)

        Args:
            frame: Input frame as numpy array
            auto_add_faces: Enable automatic face addition (background)
            similarity_threshold: Face similarity threshold
            confidence_threshold: Identification confidence threshold
            time_gap_seconds: Minimum time between similar faces
            max_faces_per_person: Maximum faces per person
            quality_check: Enable quality validation

        Returns:
            FrameProcessingResult: Immediate processing results with tagged faces
        """
        start_time = time.time()

        try:
            # IMMEDIATE PROCESSING - Step 1: Detect all faces
            face_bytes = self.image_to_bytes(frame)
            if not face_bytes:
                return FrameProcessingResult([], 0, 0, 0, time.time() - start_time, frame)

            detected_face_results = self.known_group.face_client.detect(
                face_bytes,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True,
                return_face_landmarks=True,
                return_face_attributes=['headPose', 'blur', 'exposure'],
                return_recognition_model=False
            )

            if not detected_face_results:
                return FrameProcessingResult([], 0, 0, 0, time.time() - start_time, frame)

            detected_faces = []
            background_queue_data = []
            known_faces = 0
            unknown_faces = 0

            # IMMEDIATE PROCESSING - Step 2: Fast identification and tagging (READ-ONLY)
            for i, face_result in enumerate(detected_face_results):
                face_id = face_result.face_id
                face_rect = face_result.face_rectangle
                face_attributes = face_result.face_attributes

                # Extract face crop with padding for better face capture
                x, y, w, h = face_rect.left, face_rect.top, face_rect.width, face_rect.height
                face_crop = self._extract_face_crop_with_padding(frame, x, y, w, h, padding=10)

                if face_crop is None:
                    continue

                person_id = None
                person_name = "Unknown"
                confidence = 0.0
                is_known = False
                face_color = (0, 0, 255)  # Default red for unknown
                is_identified = False

                # FAST READ-ONLY identification - known group first (higher priority)
                try:
                    known_results = self.known_group.identify_faces([str(face_id)], confidence_threshold)
                    if known_results and len(known_results) > 0:
                        result = known_results[0]
                        if result.candidates and len(result.candidates) > 0:
                            candidate = result.candidates[0]
                            if candidate.confidence >= confidence_threshold:
                                person_id = candidate.person_id
                                person_info = self.known_group.get_person_info(person_id)
                                person_name = person_info['name'] if person_info else 'Known Person'
                                confidence = candidate.confidence
                                is_known = True
                                face_color = self._get_person_color(person_name)
                                is_identified = True
                                known_faces += 1

                            # Queue for background face addition (NO immediate adding)
                            if auto_add_faces:
                                background_queue_data.append({
                                    'type': 'add_known_face',
                                    'person_id': person_id,
                                    'person_name': person_name,
                                    'face_crop': face_crop.copy(),
                                    'face_attributes': face_attributes,
                                    'similarity_threshold': similarity_threshold,
                                    'confidence_threshold': confidence_threshold,
                                    'time_gap_seconds': time_gap_seconds,
                                    'max_faces_per_person': max_faces_per_person,
                                    'quality_check': quality_check
                                })
                except Exception as e:
                    self.logger.debug(f"Known group identification error: {e}")

                # FAST READ-ONLY identification - unknown group if not identified in known
                if not is_identified:
                    try:
                        unknown_results = self.unknown_group.identify_faces([str(face_id)], confidence_threshold)
                        if unknown_results and len(unknown_results) > 0:
                            result = unknown_results[0]
                            if result.candidates and len(result.candidates) > 0:
                                candidate = result.candidates[0]
                                if candidate.confidence >= confidence_threshold:
                                    person_id = candidate.person_id
                                    person_info = self.unknown_group.get_person_info(person_id)
                                    person_name = person_info['name'] if person_info else f"person_{self.unknown_group.get_next_person_number()}"
                                    confidence = candidate.confidence
                                    is_known = False
                                    face_color = self._get_person_color(person_name)
                                    is_identified = True
                                    unknown_faces += 1

                                # Queue for background face addition (NO immediate adding)
                                if auto_add_faces:
                                    background_queue_data.append({
                                        'type': 'add_unknown_face',
                                        'person_id': person_id,
                                        'person_name': person_name,
                                        'face_crop': face_crop.copy(),
                                        'face_attributes': face_attributes,
                                        'similarity_threshold': similarity_threshold,
                                        'confidence_threshold': confidence_threshold,
                                        'time_gap_seconds': time_gap_seconds,
                                        'max_faces_per_person': max_faces_per_person,
                                        'quality_check': quality_check
                                    })
                    except Exception as e:
                        self.logger.debug(f"Unknown group identification error: {e}")

                # New unknown face if not identified anywhere
                if not is_identified:
                    next_person_number = self.unknown_group.get_next_person_number()
                    person_name = f"person_{next_person_number}"
                    confidence = 1.0
                    is_known = False
                    face_color = self._get_person_color(person_name)
                    unknown_faces += 1

                    # Queue for background person creation (NO immediate creation)
                    background_queue_data.append({
                        'type': 'create_unknown_person',
                        'person_name': person_name,
                        'face_crop': face_crop.copy(),
                        'face_attributes': face_attributes,
                        'user_data': f"New unknown person, created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        'quality_check': quality_check
                    })

                # Create detected face object for immediate response
                detected_face = DetectedFace(
                    face_id=face_id,
                    person_id=person_id,
                    person_name=person_name,
                    confidence=confidence,
                    is_known=is_known,
                    bounding_box={'x': x, 'y': y, 'width': w, 'height': h},
                    face_crop=face_crop
                )
                detected_face.face_color = face_color
                detected_faces.append(detected_face)

            # IMMEDIATE PROCESSING - Step 3: Create annotated frame
            annotated_frame = self._create_annotated_frame(frame, detected_faces)

            processing_time = time.time() - start_time
            total_faces = len(detected_faces)

            # IMMEDIATE PROCESSING - Step 4: Return result immediately
            result = FrameProcessingResult(
                detected_faces=detected_faces,
                total_faces=total_faces,
                known_faces=known_faces,
                unknown_faces=unknown_faces,
                processing_time=processing_time,
                annotated_frame=annotated_frame
            )

            # BACKGROUND PROCESSING - Step 5: Queue heavy operations
            if background_queue_data:
                self._add_to_processing_queue(background_queue_data)
                self.logger.debug(f"Queued {len(background_queue_data)} items for background processing")

            # Optional: Save Annotated frame in background
            if self.save_processed_frames:
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                frame_save_data = {
                    'type': 'save_processed_frame',
                    'frame': annotated_frame.copy(), 
                    'frame_id': f"frame_{timestamp}"
                }
                self._add_to_processing_queue([frame_save_data])

            self.logger.info(f"Processed: {total_faces} faces, {known_faces} known, {unknown_faces} unknown, {processing_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"Error in process_frame: {e}")
            processing_time = time.time() - start_time
            return FrameProcessingResult([], 0, 0, 0, processing_time, frame.copy())

    def _check_face_quality(self, face_attributes, quality_check: bool) -> bool:
        """
        Check if face meets quality criteria for training

        Criteria:
        - Good head pose (not too tilted)
        - Low blur level
        - Good exposure
        """
        if not quality_check or not face_attributes:
            return True

        try:
            # Check head pose - should be relatively straight
            if hasattr(face_attributes, 'head_pose') and isinstance(face_attributes.head_pose, HeadPose):
                hp = face_attributes.head_pose
                if any(abs(getattr(hp, attr, 0)) > MAX_HEAD_POSE_ANGLE for attr in ['pitch', 'yaw', 'roll']):
                    self.logger.debug("Face rejected: poor head pose (pitch/yaw/roll too large)")
                    return False

            # Check blur level - should be low
            if hasattr(face_attributes, 'blur') and isinstance(face_attributes.blur, BlurProperties):
                blur = face_attributes.blur
                try:
                    if blur and blur.blur_level in [BlurLevel.medium, BlurLevel.high]:
                        self.logger.debug(f"Face rejected: too blurry (blur level = {blur.blur_level})")
                        return False
                except (ValueError, TypeError):
                    self.logger.debug("Face quality check: blur level not numeric, accepting")

            # Check exposure - should be good
            if hasattr(face_attributes, 'exposure') and isinstance(face_attributes.exposure, ExposureProperties):
                exposure = face_attributes.exposure
                try:
                    if exposure and exposure.exposure_level != ExposureLevel.good_exposure:
                        self.logger.debug(f"Face rejected: poor exposure (exposure level = {exposure.exposure_level})")
                        return False

                except (ValueError, TypeError):
                    self.logger.debug("Face quality check: exposure level not numeric, accepting")

            return True

        except Exception as e:
            self.logger.warning(f"Error checking face quality: {e}")
            return True  # Default to accepting if quality check fails

    def _get_person_color(self, person_name: str) -> tuple:
        """Get a consistent color for each person name"""
        # Predefined colors for different persons
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Light Green
            (255, 192, 203) # Pink
        ]

        # Use hash of name to get consistent color
        color_index = hash(person_name) % len(colors)
        return colors[color_index]

    def _create_annotated_frame(self, frame: np.ndarray, detected_faces: List[DetectedFace]) -> np.ndarray:
        """Create annotated frame with different colors for different persons"""
        import cv2

        annotated_frame = frame.copy()

        for face in detected_faces:
            bbox = face.bounding_box
            color = getattr(face, 'face_color', (0, 0, 255))  # Default red if no color set

            # Draw rectangle
            cv2.rectangle(annotated_frame, (bbox['x'], bbox['y']),
                        (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), color, 2)

            # Draw person name and confidence
            label = f"{face.person_name} ({face.confidence:.2f})"
            cv2.putText(annotated_frame, label,
                      (bbox['x'], bbox['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return annotated_frame

    def _coordinate_training(self):
        """
        Coordinate training between known and unknown groups

        Flow:
        - Check if known group is training
        - If not training, trigger training with any pending unknown data
        - If training, set flag for unknown group
        """
        try:
            # Check known group training status
            known_training_status = self.known_group.get_training_status()

            if known_training_status != "running":
                # Known group not training - check for pending unknown data
                pending_data = self.unknown_group.get_pending_training_data()

                if pending_data:
                    self.logger.info("Transferring pending unknown data to known group before training")
                    # Transfer pending data would be handled by the mapping function

                # Trigger known group training
                self.known_group.train_group()
                self.logger.info("Triggered known group training")

                # Also trigger unknown group training
                self.unknown_group.train_group()
                self.logger.info("Triggered unknown group training")
            else:
                # Known group is training - alert unknown group
                self.unknown_group.set_known_group_training_flag(True)
                self.logger.info("Known group is training - flagged unknown group")

        except Exception as e:
            self.logger.error(f"Error coordinating training: {e}")

    def image_to_bytes(self, image: np.ndarray) -> Optional[bytes]:
        """Convert numpy image to bytes for Azure Face API"""
        try:
            import cv2
            _, buffer = cv2.imencode('.jpg', image)
            return buffer.tobytes()
        except Exception as e:
            self.logger.error(f"Error converting image to bytes: {e}")
            return None

    def get_known_group_id(self) -> str:
        """Get the known group ID"""
        return self.known_group.get_group_id()
    
    def get_unknown_group_id(self) -> str:
        """Get the unknown group ID"""
        return self.unknown_group.get_group_id()
    
    def create_known_person(self, name: str, user_data: str = "") -> Optional[str]:
        """Create a new known person"""
        return self.known_group.add_person(name, user_data)

    def add_known_person(self, person_name: str, face_image: np.ndarray) -> Optional[str]:
        """Add a known person with face image"""
        person_id = self.known_group.add_person(person_name)
        if person_id:
            if self.known_group.add_face_to_person(person_id, face_image):
                self.known_group.train_group()
                return person_id
            else:
                self.known_group.delete_person(person_id)
        return None

    def add_face_to_known_person(self, person_id: str, face_image: np.ndarray, user_data: str = "") -> Optional[str]:
        """Add face image to existing known person"""
        try:
            success = self.known_group.add_face_to_person(person_id, face_image, user_data)
            if success:
                self.logger.info(f"Successfully added face to known person {person_id}")
                return "face_added"  # Return success indicator
            else:
                self.logger.error(f"Failed to add face to known person {person_id}")
                return None
        except Exception as e:
            self.logger.error(f"Error adding face to known person {person_id}: {e}")
            return None
    
    def map_unknown_person_to_known(self, unknown_person_name: str, known_name: str, user_data: str = "") -> bool:
        """
        Map an unknown person to known group with proper name and user_data

        This is the key function for step 5 of your workflow:
        - Takes person_#unknown_number from unknown group
        - Maps to proper name (not person_1, person_2, etc.)
        - Transfers all faces and data to known group
        - Triggers training coordination

        Args:
            unknown_person_name: The person_X name from unknown group (e.g., "person_1")
            known_name: The actual name to use in known group (e.g., "Dad", "John")
            user_data: Additional user data for the person

        Returns:
            bool: True if mapping successful, False otherwise
        """
        try:
            self.logger.info(f"Mapping unknown person '{unknown_person_name}' to known person '{known_name}'")

            # Get unknown person data
            unknown_person_data = self.unknown_group.get_person_data_for_transfer(unknown_person_name)
            if not unknown_person_data:
                self.logger.error(f"Unknown person '{unknown_person_name}' not found")
                return False

            # Check known group training status
            known_training_status = self.known_group.get_training_status()

            if known_training_status == "running":
                # Known group is training - alert and flag
                self.logger.warning("Known group is currently training - flagging for later transfer")
                self.unknown_group.set_known_group_training_flag(True)
                # Mark this person for pending transfer
                self.unknown_group.mark_person_for_transfer(unknown_person_name, known_name, user_data)
                return True

            # Create person in known group
            known_person_id = self.known_group.add_person(known_name, user_data)
            if not known_person_id:
                self.logger.error(f"Failed to create known person: {known_name}")
                return False

            # Transfer all faces from unknown to known
            faces_transferred = 0
            for face_data in unknown_person_data['faces']:
                if self.known_group.add_face_to_person(known_person_id, face_data['image'], face_data.get('user_data', '')):
                    faces_transferred += 1

            self.logger.info(f"Transferred {faces_transferred} faces from {unknown_person_name} to {known_name}")

            # Mark unknown person as transferred
            self.unknown_group.mark_person_as_transferred(unknown_person_name)

            # Trigger training coordination
            self._coordinate_training_after_transfer()

            return True

        except Exception as e:
            self.logger.error(f"Error mapping unknown person to known: {e}")
            return False

    def _coordinate_training_after_transfer(self):
        """
        Coordinate training after transferring data from unknown to known

        Flow from step 6 of your workflow:
        - Check if known group is training
        - If not training, trigger training
        - If training, wait and retry
        - After training, check for more pending transfers
        """
        try:
            known_training_status = self.known_group.get_training_status()

            if known_training_status != "running":
                # Check if there are more pending transfers
                more_pending = self.unknown_group.has_pending_transfers()

                if more_pending:
                    self.logger.info("More pending transfers found - appending to training data")
                    # Process pending transfers would be handled here

                # Trigger known group training
                self.known_group.train_group()
                self.logger.info("Triggered known group training after transfer")

                # Set up post-training cleanup
                self._schedule_post_training_cleanup()
            else:
                self.logger.info("Known group still training - transfer completed, will train after current training")

        except Exception as e:
            self.logger.error(f"Error coordinating training after transfer: {e}")

    def _schedule_post_training_cleanup(self):
        """
        Schedule cleanup of transferred persons from unknown group

        From step 7 of your workflow:
        - After training completes, mark transferred data as trained
        - Delete persons and images from unknown group
        - Keep unknown group clean and minimal
        """
        try:
            # This would typically be called after training completion
            # For now, we'll mark it for cleanup
            self.unknown_group.schedule_cleanup_after_training()
            self.logger.info("Scheduled post-training cleanup for unknown group")

        except Exception as e:
            self.logger.error(f"Error scheduling post-training cleanup: {e}")

    def cleanup_transferred_persons(self):
        """
        Clean up persons that have been successfully transferred and trained

        This implements step 7 of your workflow:
        - Check training status
        - Delete transferred persons from unknown group
        - Keep unknown group minimal
        """
        try:
            # Check if training is complete
            known_training_status = self.known_group.get_training_status()

            if known_training_status == "succeeded":
                # Training succeeded - clean up transferred persons
                cleaned_count = self.unknown_group.cleanup_transferred_persons()
                self.logger.info(f"Cleaned up {cleaned_count} transferred persons from unknown group")
                return True
            else:
                self.logger.info(f"Known group training status: {known_training_status} - cleanup postponed")
                return False

        except Exception as e:
            self.logger.error(f"Error cleaning up transferred persons: {e}")
            return False

    def train_all_groups(self):
        """Train both known and unknown groups with coordination"""
        self._coordinate_training()

    # Queue Processing System for Background Operations

    def _add_to_processing_queue(self, face_queue_data: List[Dict[str, Any]]):
        """Add face data to processing queue for background operations"""
        try:
            queue_item = {
                'timestamp': datetime.now(),
                'face_data': face_queue_data,
                'operation': 'face_processing'
            }
            self.processing_queue.put(queue_item)
            self.logger.info(f"Added {len(face_queue_data)} faces to processing queue")

            # Start queue processor if not running
            if not self.queue_processing_active:
                self.start_queue_processor()

        except Exception as e:
            self.logger.error(f"Error adding to processing queue: {e}")

    def start_queue_processor(self):
        """Start the background queue processor thread"""
        if self.queue_processing_active:
            self.logger.info("Queue processor already running")
            return

        try:
            self.queue_processing_active = True
            self.queue_processor_thread = threading.Thread(
                target=self._process_queue_worker,
                daemon=True,
                name="FaceProcessingQueue"
            )
            self.queue_processor_thread.start()
            self.logger.info("Started background queue processor")

        except Exception as e:
            self.logger.error(f"Error starting queue processor: {e}")
            self.queue_processing_active = False

    def stop_queue_processor(self):
        """Stop the background queue processor"""
        try:
            self.queue_processing_active = False
            if self.queue_processor_thread and self.queue_processor_thread.is_alive():
                self.queue_processor_thread.join(timeout=5.0)
            self.logger.info("Stopped background queue processor")

        except Exception as e:
            self.logger.error(f"Error stopping queue processor: {e}")

    def _process_queue_worker(self):
        """Background worker that processes queued face data"""
        self.logger.info("Queue processor worker started")

        while self.queue_processing_active:
            try:
                # Get item from queue with timeout
                queue_item = self.processing_queue.get(timeout=1.0)

                if queue_item['operation'] == 'face_processing':
                    self._process_queued_faces(queue_item['face_data'])

                # Mark task as done
                self.processing_queue.task_done()

            except queue.Empty:
                # No items in queue, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in queue processor worker: {e}")

        self.logger.info("Queue processor worker stopped")

    def _process_queued_faces(self, face_data_list: List[Dict[str, Any]]):
        """
        Process queued face data with FIFO order and efficient training triggers

        FIFO Processing: All operations execute in exact order they were added
        Training Efficiency: Only trigger training when consecutive face addition operations end
        """
        try:
            self.logger.info(f"Processing {len(face_data_list)} queued items in FIFO order")

            # Track consecutive face addition operations for training optimization
            consecutive_known_face_ops = 0
            consecutive_unknown_face_ops = 0

            # Process operations in FIFO order (exact order they were added)
            for i, face_data in enumerate(face_data_list):
                try:
                    item_type = face_data.get('type', 'legacy')
                    self.logger.debug(f"Processing queue item {i+1}/{len(face_data_list)}: {item_type}")

                    # Track consecutive face addition operations
                    if item_type in ['add_known_face', 'transfer_to_known']:
                        consecutive_known_face_ops += 1
                        consecutive_unknown_face_ops = 0  # Reset unknown counter
                    elif item_type in ['add_unknown_face', 'create_unknown_person']:
                        consecutive_unknown_face_ops += 1
                        consecutive_known_face_ops = 0  # Reset known counter
                    else:
                        # Different operation type - trigger training for any pending consecutive operations
                        if consecutive_known_face_ops > 0:
                            self.logger.info(f"Triggering training for known group after {consecutive_known_face_ops} consecutive operations")
                            self._trigger_group_training('known')
                            consecutive_known_face_ops = 0

                        if consecutive_unknown_face_ops > 0:
                            self.logger.info(f"Triggering training for unknown group after {consecutive_unknown_face_ops} consecutive operations")
                            self._trigger_group_training('unknown')
                            consecutive_unknown_face_ops = 0

                    # Process the current operation
                    if item_type == 'add_known_face':
                        self._process_add_known_face_background(face_data)
                    elif item_type == 'transfer_to_known':
                        self._process_transfer_to_known_background(face_data)
                    elif item_type == 'add_unknown_face':
                        self._process_add_unknown_face_background(face_data)
                    elif item_type == 'create_unknown_person':
                        self._process_create_unknown_person_background(face_data)
                    elif item_type == 'save_processed_frame':
                        self._process_save_frame_background(face_data)
                    elif item_type == 'cleanup_unknown_person':
                        self._process_cleanup_unknown_person_background(face_data)
                    else:
                        # Legacy support for old queue items
                        self._process_legacy_queue_item(face_data)

                except Exception as e:
                    self.logger.error(f"Error processing queue item {i+1}: {e}")

            # Trigger training for any remaining consecutive operations at the end of batch
            if consecutive_known_face_ops > 0:
                self.logger.info(f"Triggering training for known group after {consecutive_known_face_ops} consecutive operations (batch complete)")
                self._trigger_group_training('known')

            if consecutive_unknown_face_ops > 0:
                self.logger.info(f"Triggering training for unknown group after {consecutive_unknown_face_ops} consecutive operations (batch complete)")
                self._trigger_group_training('unknown')

        except Exception as e:
            self.logger.error(f"Error processing queued faces: {e}")

    def _process_legacy_queue_item(self, face_data: Dict[str, Any]):
        """Process legacy queue items for backward compatibility"""
        try:
            face_crop = face_data.get('face_crop')
            person_name = face_data.get('person_name')
            is_known = face_data.get('is_known')
            auto_add_faces = face_data.get('auto_add_faces', True)

            if not auto_add_faces or face_crop is None:
                return

            # Check face quality
            quality_check = face_data.get('quality_check', True)
            if quality_check and not self._check_face_quality_from_crop(face_crop):
                self.logger.debug(f"Skipping face {person_name} - poor quality")
                return

            # Process based on known/unknown status
            if is_known:
                self._process_known_face_background(face_data)
            else:
                self._process_unknown_face_background(face_data)

        except Exception as e:
            self.logger.error(f"Error processing legacy queue item: {e}")

    def _process_add_known_face_background(self, face_data: Dict[str, Any]):
        """Process adding face to known person in background"""
        try:
            person_id = face_data.get('person_id')
            person_name = face_data.get('person_name')
            face_crop = face_data.get('face_crop')
            face_attributes = face_data.get('face_attributes')

            # Quality check
            quality_check = face_data.get('quality_check', True)
            if quality_check and not self._check_face_quality(face_attributes, quality_check):
                self.logger.debug(f"Background: Skipping known face {person_name} - poor quality")
                return

            # Add face with similarity check
            added = self.known_group.add_face_with_similarity_check(
                face_crop, person_id,
                face_data.get('similarity_threshold', 0.8),
                face_data.get('confidence_threshold', 0.8),
                face_data.get('time_gap_seconds', 0),
                face_data.get('max_faces_per_person', 5)
            )

            if added:
                self.logger.info(f"Added face to known person {person_name}")
                # Face crop is automatically saved by the known_face_group when face is successfully added
            else:
                self.logger.debug(f"Face not added to known person {person_name} (similarity/quality check failed)")

        except Exception as e:
            self.logger.error(f"Error adding known face: {e}")
            import traceback
            self.logger.error(f"Processing traceback: {traceback.format_exc()}")

    def _process_add_unknown_face_background(self, face_data: Dict[str, Any]):
        """Process adding face to unknown person in background"""
        try:
            person_id = face_data.get('person_id')
            person_name = face_data.get('person_name')
            face_crop = face_data.get('face_crop')
            face_attributes = face_data.get('face_attributes')

            # Quality check
            quality_check = face_data.get('quality_check', True)
            if quality_check and not self._check_face_quality(face_attributes, quality_check):
                self.logger.debug(f"Background: Skipping unknown face {person_name} - poor quality")
                return

            # Add face with similarity check
            added = self.unknown_group.add_face_with_similarity_check(
                face_crop, person_id,
                face_data.get('similarity_threshold', 0.8),
                face_data.get('confidence_threshold', 0.8),
                face_data.get('time_gap_seconds', 0),
                face_data.get('max_faces_per_person', 5)
            )

            if added:
                self.logger.info(f"Added face to unknown person {person_name}")
                # Face crop is automatically saved by the unknown_face_group when face is successfully added
            else:
                self.logger.debug(f"Face not added to unknown person {person_name} (similarity/quality check failed)")

        except Exception as e:
            self.logger.error(f"Error adding unknown face: {e}")
            import traceback
            self.logger.error(f"Processing traceback: {traceback.format_exc()}")

    def _process_create_unknown_person_background(self, face_data: Dict[str, Any]):
        """Process creating new unknown person in background"""
        try:
            person_name = face_data.get('person_name')
            face_crop = face_data.get('face_crop')
            face_attributes = face_data.get('face_attributes')
            user_data = face_data.get('user_data', '')

            # Quality check
            quality_check = face_data.get('quality_check', True)
            if quality_check and not self._check_face_quality(face_attributes, quality_check):
                self.logger.debug(f"Background: Skipping new unknown person {person_name} - poor quality")
                return

            # Create new unknown person
            new_person_id = self.unknown_group.add_unknown_person(face_crop, user_data)

            if new_person_id:
                self.logger.info(f"Created new unknown person {person_name}")
                # Face crop is automatically saved by the unknown_face_group when person is successfully created

        except Exception as e:
            self.logger.error(f"Error creating unknown person: {e}")
            import traceback
            self.logger.error(f"Processing traceback: {traceback.format_exc()}")

    def _process_save_frame_background(self, frame_data: Dict[str, Any]):
        """Process saving processed frame in background"""
        try:
            frame = frame_data.get('frame')
            frame_id = frame_data.get('frame_id')

            if frame is not None and frame_id:
                frame_path = self.save_processed_frame(frame, frame_id)
                if frame_path:
                    self.logger.debug(f"Background: Saved processed frame to {frame_path}")

        except Exception as e:
            self.logger.error(f"Error saving frame in background: {e}")

    def _process_transfer_to_known_background(self, transfer_data: Dict[str, Any]):
        """Process transfer of unknown person to known group via queue"""
        try:
            unknown_person_id = transfer_data.get('unknown_person_id')
            known_person_name = transfer_data.get('known_person_name')
            known_user_data = transfer_data.get('known_user_data', '')
            face_images = transfer_data.get('face_images', [])

            self.logger.info(f"Queue: Transferring unknown person {unknown_person_id} to known as {known_person_name}")

            # Create known person
            known_person_id = self.known_group.add_person(known_person_name, known_user_data)
            if not known_person_id:
                self.logger.error(f"Failed to create known person {known_person_name}")
                return

            # Add all face images to known person
            added_count = 0
            for face_image in face_images:
                try:
                    # The known group's add_face_to_person method automatically handles saving
                    # to the structured file system, so no need for duplicate save call
                    success = self.known_group.add_face_to_person(known_person_id, face_image)
                    if success:
                        added_count += 1
                        self.logger.debug(f"Successfully added and saved face for {known_person_name}")
                except Exception as e:
                    self.logger.error(f"Error adding face to known person: {e}")

            self.logger.info(f"Queue: Added {added_count} faces to known person {known_person_name}")

            # Queue cleanup of unknown person after successful transfer
            cleanup_data = {
                'type': 'cleanup_unknown_person',
                'unknown_person_id': unknown_person_id,
                'transfer_completed': True
            }
            self._add_to_processing_queue([cleanup_data])

        except Exception as e:
            self.logger.error(f"Error transferring to known group in background: {e}")

    def _process_cleanup_unknown_person_background(self, cleanup_data: Dict[str, Any]):
        """Process cleanup of unknown person after transfer via queue"""
        try:
            unknown_person_id = cleanup_data.get('unknown_person_id')
            transfer_completed = cleanup_data.get('transfer_completed', False)

            if transfer_completed:
                self.logger.info(f"Queue: Cleaning up unknown person {unknown_person_id} after successful transfer")

                # Delete unknown person from Azure
                success = self.unknown_group.delete_person(unknown_person_id)
                if success:
                    # Clean up structured file system
                    self._cleanup_person_files(unknown_person_id, False)
                    self.logger.info(f"Queue: Successfully cleaned up unknown person {unknown_person_id}")

                    # Trigger training after person deletion
                    self.logger.info("Queue: Starting unknown group training after person deletion")
                    training_success = self.unknown_group.train_group()
                    if training_success:
                        self.logger.info("Queue: ✅ Unknown group training completed after cleanup")
                    else:
                        self.logger.warning("Queue: ❌ Unknown group training failed after cleanup")
                else:
                    self.logger.error(f"Queue: Failed to delete unknown person {unknown_person_id}")

        except Exception as e:
            self.logger.error(f"Error cleaning up unknown person in background: {e}")

    def _cleanup_person_files(self, person_id: str, is_known: bool):
        """Clean up person folder and files from saved_images directory"""
        try:
            import shutil

            # Determine the group directory
            group_type = "known" if is_known else "unknown"
            person_folder = self.base_save_directory / group_type / person_id

            if person_folder.exists():
                # Remove the entire person folder and all its contents
                shutil.rmtree(person_folder)
                self.logger.info(f"Successfully removed {group_type} person folder: {person_id}")
            else:
                self.logger.debug(f"Person folder does not exist: {person_folder}")

        except Exception as e:
            self.logger.error(f"Error cleaning up person files for {person_id}: {e}")

    def _trigger_group_training(self, group_type: str):
        """Trigger training for specific group with efficient batching"""
        try:
            if group_type == 'known':
                self.logger.info("Triggering training for known group")
                training_success = self.known_group.train_group()
                if training_success:
                    self.logger.info("Known group training initiated successfully")
                else:
                    self.logger.error("Failed to initiate known group training")

            elif group_type == 'unknown':
                self.logger.info("Triggering training for unknown group")
                training_success = self.unknown_group.train_group()
                if training_success:
                    self.logger.info("Unknown group training initiated successfully")
                else:
                    self.logger.error("Failed to initiate unknown group training")

        except Exception as e:
            self.logger.error(f"Error triggering training for {group_type} group: {e}")

    def queue_person_transfer(self, unknown_person_id: str, known_person_name: str,
                             known_user_data: str = "") -> bool:
        """Queue transfer of unknown person to known group via queue system"""
        try:
            # Get unknown person's face images
            face_images = self.unknown_group.get_person_face_images(unknown_person_id)
            if not face_images:
                self.logger.error(f"No face images found for unknown person {unknown_person_id}")
                return False

            # Queue transfer operation
            transfer_data = {
                'type': 'transfer_to_known',
                'unknown_person_id': unknown_person_id,
                'known_person_name': known_person_name,
                'known_user_data': known_user_data,
                'face_images': face_images
            }

            self._add_to_processing_queue([transfer_data])
            self.logger.info(f"Queued transfer of unknown person {unknown_person_id} to known as {known_person_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error queuing person transfer: {e}")
            return False

    def _process_known_face_background(self, face_data: Dict[str, Any]):
        """Process known face in background"""
        try:
            person_id = face_data.get('person_id')
            person_name = face_data.get('person_name')
            face_crop = face_data.get('face_crop')
            user_data = face_data.get('user_data', '')
            similarity_threshold = face_data.get('similarity_threshold', 0.8)
            confidence_threshold = face_data.get('confidence_threshold', 0.8)
            time_gap_seconds = face_data.get('time_gap_seconds', 0)
            max_faces_per_person = face_data.get('max_faces_per_person', 5)

            if person_id and face_crop is not None:
                added = self.known_group.add_face_with_similarity_check(
                    face_crop, person_id, similarity_threshold, confidence_threshold,
                    time_gap_seconds, max_faces_per_person
                )
                if added:
                    self.logger.info(f"Background: Added face to known person {person_name}")

                    # Save face crop to structured file system
                    saved_path = self.save_person_face_crop(
                        person_id, person_name, face_crop, is_known=True, user_data=user_data
                    )
                    if saved_path:
                        self.logger.debug(f"Saved known face crop: {saved_path}")

        except Exception as e:
            self.logger.error(f"Error processing known face in background: {e}")

    def _process_unknown_face_background(self, face_data: Dict[str, Any]):
        """Process unknown face in background"""
        try:
            person_id = face_data.get('person_id')
            face_crop = face_data.get('face_crop')
            person_name = face_data.get('person_name')

            if face_crop is not None:
                if person_id:
                    # Existing unknown person - add face
                    similarity_threshold = face_data.get('similarity_threshold', 0.8)
                    confidence_threshold = face_data.get('confidence_threshold', 0.8)
                    time_gap_seconds = face_data.get('time_gap_seconds', 0)
                    max_faces_per_person = face_data.get('max_faces_per_person', 5)

                    added = self.unknown_group.add_face_with_similarity_check(
                        face_crop, person_id, similarity_threshold, confidence_threshold,
                        time_gap_seconds, max_faces_per_person
                    )
                    if added:
                        self.logger.info(f"Background: Added face to unknown person {person_name}")
                else:
                    # New unknown person - create
                    new_person_id = self.unknown_group.add_unknown_person(face_crop)
                    if new_person_id:
                        self.logger.info(f"Background: Created new unknown person {person_name}")

        except Exception as e:
            self.logger.error(f"Error processing unknown face in background: {e}")

    def _check_face_quality_from_crop(self, face_crop: np.ndarray) -> bool:
        """Check face quality from crop image (simplified version)"""
        try:
            # Basic quality checks on the crop
            if face_crop is None or face_crop.size == 0:
                return False

            # Check if image is too small
            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                return False

            # Check if image is too blurry (using Laplacian variance)
            import cv2
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var < 100:  # Threshold for blur detection
                return False

            return True

        except Exception as e:
            self.logger.warning(f"Error checking face quality from crop: {e}")
            return True  # Default to accepting if check fails

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue processing status"""
        return {
            'queue_size': self.processing_queue.qsize(),
            'processor_active': self.queue_processing_active,
            'processor_thread_alive': self.queue_processor_thread.is_alive() if self.queue_processor_thread else False
        }

    def process_queue_manually(self):
        """Manually process all items in queue (like a cron job)"""
        try:
            processed_count = 0

            while not self.processing_queue.empty():
                try:
                    queue_item = self.processing_queue.get_nowait()

                    if queue_item['operation'] == 'face_processing':
                        self._process_queued_faces(queue_item['face_data'])
                        processed_count += 1

                    self.processing_queue.task_done()

                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error in manual queue processing: {e}")

            self.logger.info(f"Manual queue processing completed: {processed_count} items processed")
            return processed_count

        except Exception as e:
            self.logger.error(f"Error in manual queue processing: {e}")
            return 0

    # File System Management Methods

    def _initialize_directory_structure(self):
        """Initialize the structured directory system for saving face data"""
        try:
            # Create base directory structure
            directories = [
                self.base_save_directory,
                self.base_save_directory / "unknown",
                self.base_save_directory / "known"
            ]

            # Add frame_processed directory if enabled
            if self.save_processed_frames:
                directories.append(self.base_save_directory / "frame_processed")

            # Create all directories
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Ensured directory exists: {directory}")

            self.logger.info("Directory structure initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing directory structure: {e}")

    def _get_person_directory(self, person_id: str, is_known: bool) -> Path:
        """Get the directory path for a specific person"""
        group_dir = "known" if is_known else "unknown"
        return self.base_save_directory / group_dir / person_id

    def _ensure_person_directory(self, person_id: str, is_known: bool) -> Path:
        """Ensure person directory exists and return the path"""
        person_dir = self._get_person_directory(person_id, is_known)
        person_dir.mkdir(parents=True, exist_ok=True)
        return person_dir

    def _generate_timestamp_filename(self, person_name: str, extension: str = "jpg") -> str:
        """Generate filename with timestamp format: {person_name}_{YYYY-MM-DD_HH-MM-SS}.{extension}"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{person_name}_{timestamp}.{extension}"

    def _get_unique_filename(self, directory: Path, base_filename: str) -> str:
        """Get unique filename by adding incremental suffix if file exists"""
        filepath = directory / base_filename

        if not filepath.exists():
            return base_filename

        # Extract name and extension
        name_part = filepath.stem
        extension = filepath.suffix

        # Add incremental suffix
        counter = 1
        while True:
            new_filename = f"{name_part}_{counter:03d}{extension}"
            new_filepath = directory / new_filename
            if not new_filepath.exists():
                return new_filename
            counter += 1

    def _extract_face_crop_with_padding(self, frame: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 10) -> Optional[np.ndarray]:
        """
        Extract face crop with padding around the detected face bounding box

        Args:
            frame: Original frame image
            x, y, w, h: Face bounding box coordinates
            padding: Padding in pixels to add around the face (default: 10)

        Returns:
            Face crop with padding, or None if extraction fails
        """
        try:
            frame_height, frame_width = frame.shape[:2]

            # Calculate padded coordinates
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame_width, x + w + padding)
            y_end = min(frame_height, y + h + padding)

            # Extract face crop with padding
            face_crop = frame[y_start:y_end, x_start:x_end]

            # Validate crop
            if face_crop.size == 0:
                self.logger.warning(f"Empty face crop extracted at ({x}, {y}, {w}, {h}) with padding {padding}")
                return None

            return face_crop

        except Exception as e:
            self.logger.error(f"Error extracting face crop with padding: {e}")
            return None

    def save_person_face_crop(self, person_id: str, person_name: str, face_crop: np.ndarray,
                             is_known: bool, user_data: str = "") -> Optional[str]:
        """
        Save face crop image for a person with structured file system

        Args:
            person_id: Unique identifier for the person
            person_name: Display name of the person
            face_crop: Face crop image as numpy array
            is_known: Whether person belongs to known or unknown group
            user_data: Additional information about the person

        Returns:
            str: Path to saved image file, None if failed
        """
        try:
            with self.file_lock:  # Thread-safe file operations
                # Ensure person directory exists
                person_dir = self._ensure_person_directory(person_id, is_known)

                # Generate filename with timestamp
                base_filename = self._generate_timestamp_filename(person_name)
                unique_filename = self._get_unique_filename(person_dir, base_filename)

                # Save face crop image
                image_path = person_dir / unique_filename
                success = cv2.imwrite(str(image_path), face_crop)

                if success:
                    self.logger.info(f"Saved face crop: {image_path}")

                    # Update or create person_data.json
                    self._update_person_data_json(person_dir, person_id, person_name, user_data)

                    return str(image_path)
                else:
                    self.logger.error(f"Failed to save face crop: {image_path}")
                    return None

        except Exception as e:
            self.logger.error(f"Error saving person face crop: {e}")
            return None

    def _update_person_data_json(self, person_dir: Path, person_id: str,
                                person_name: str, user_data: str):
        """Update or create person_data.json file"""
        try:
            json_path = person_dir / "person_data.json"

            # Load existing data or create new
            person_data = {}
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        person_data = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid JSON in {json_path}, creating new")
                    person_data = {}

            # Update person data
            person_data.update({
                "name": person_name,
                "person_id": person_id,
                "user_data": user_data,
                "last_updated": datetime.now().isoformat(),
                "face_count": len([f for f in person_dir.glob("*.jpg") if f.name != "person_data.json"])
            })

            # Save updated data
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(person_data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Updated person data JSON: {json_path}")

        except Exception as e:
            self.logger.error(f"Error updating person data JSON: {e}")

    def save_processed_frame(self, frame: np.ndarray, frame_id: str = None) -> Optional[str]:
        """
        Save processed frame with annotations

        Args:
            frame: Annotated frame as numpy array
            frame_id: Optional frame identifier

        Returns:
            str: Path to saved frame, None if disabled or failed
        """
        if not self.save_processed_frames:
            return None

        try:
            with self.file_lock:
                frame_dir = self.base_save_directory / "frame_processed"
                frame_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename
                if frame_id:
                    base_filename = f"{frame_id}.jpg"
                else:
                    base_filename = f"frame_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

                unique_filename = self._get_unique_filename(frame_dir, base_filename)
                frame_path = frame_dir / unique_filename

                # Save frame
                success = cv2.imwrite(str(frame_path), frame)

                if success:
                    self.logger.debug(f"Saved processed frame: {frame_path}")
                    return str(frame_path)
                else:
                    self.logger.error(f"Failed to save processed frame: {frame_path}")
                    return None

        except Exception as e:
            self.logger.error(f"Error saving processed frame: {e}")
            return None

    def get_person_data(self, person_id: str, is_known: bool) -> Optional[Dict[str, Any]]:
        """
        Get person data from JSON file

        Args:
            person_id: Person identifier
            is_known: Whether to look in known or unknown group

        Returns:
            dict: Person data or None if not found
        """
        try:
            person_dir = self._get_person_directory(person_id, is_known)
            json_path = person_dir / "person_data.json"

            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error reading person data: {e}")
            return None

    def get_person_face_images(self, person_id: str, is_known: bool) -> List[str]:
        """
        Get list of face image paths for a person

        Args:
            person_id: Person identifier
            is_known: Whether to look in known or unknown group

        Returns:
            list: List of image file paths
        """
        try:
            person_dir = self._get_person_directory(person_id, is_known)

            if person_dir.exists():
                # Get all jpg files except person_data.json
                image_files = [str(f) for f in person_dir.glob("*.jpg")]
                return sorted(image_files)
            else:
                return []

        except Exception as e:
            self.logger.error(f"Error getting person face images: {e}")
            return []

    def list_all_persons(self, is_known: bool) -> List[Dict[str, Any]]:
        """
        List all persons in known or unknown group with their data

        Args:
            is_known: Whether to list known or unknown persons

        Returns:
            list: List of person data dictionaries
        """
        try:
            group_dir = self.base_save_directory / ("known" if is_known else "unknown")
            persons = []

            if group_dir.exists():
                for person_dir in group_dir.iterdir():
                    if person_dir.is_dir():
                        person_id = person_dir.name
                        person_data = self.get_person_data(person_id, is_known)

                        if person_data:
                            person_data['image_count'] = len(self.get_person_face_images(person_id, is_known))
                            persons.append(person_data)
                        else:
                            # Create basic data if JSON doesn't exist
                            image_count = len(self.get_person_face_images(person_id, is_known))
                            persons.append({
                                'person_id': person_id,
                                'name': person_id,
                                'user_data': '',
                                'image_count': image_count
                            })

            return persons

        except Exception as e:
            self.logger.error(f"Error listing persons: {e}")
            return []

    def train_from_structured_data(self, is_known: bool = True) -> bool:
        """
        Train face groups using structured file system data

        Args:
            is_known: Whether to train known or unknown group

        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Training {'known' if is_known else 'unknown'} group from structured data")

            # Get all persons from file system
            persons = self.list_all_persons(is_known)

            if not persons:
                self.logger.warning(f"No persons found in {'known' if is_known else 'unknown'} group")
                return True

            success_count = 0
            total_count = len(persons)

            for person_data in persons:
                try:
                    person_id = person_data['person_id']
                    person_name = person_data['name']
                    user_data = person_data.get('user_data', '')

                    # Get face images for this person
                    image_paths = self.get_person_face_images(person_id, is_known)

                    if not image_paths:
                        self.logger.warning(f"No images found for person {person_name}")
                        continue

                    # Load images
                    face_images = []
                    for image_path in image_paths:
                        image = cv2.imread(image_path)
                        if image is not None:
                            face_images.append(image)
                        else:
                            self.logger.warning(f"Failed to load image: {image_path}")

                    if not face_images:
                        self.logger.warning(f"No valid images loaded for person {person_name}")
                        continue

                    # Add person to appropriate group
                    if is_known:
                        # Add to known group
                        for i, face_image in enumerate(face_images):
                            if i == 0:
                                # Create person with first image
                                created_person_id = self.known_group.add_person(person_name, face_image, user_data)
                                if created_person_id:
                                    self.logger.info(f"Created known person: {person_name}")
                                else:
                                    self.logger.error(f"Failed to create known person: {person_name}")
                                    break
                            else:
                                # Add additional faces
                                if created_person_id:
                                    added = self.known_group.add_face_to_person(created_person_id, face_image)
                                    if added:
                                        self.logger.debug(f"Added face {i+1} to {person_name}")
                                    else:
                                        self.logger.warning(f"Failed to add face {i+1} to {person_name}")
                    else:
                        # Add to unknown group
                        for i, face_image in enumerate(face_images):
                            if i == 0:
                                # Create person with first image
                                created_person_id = self.unknown_group.add_unknown_person(face_image)
                                if created_person_id:
                                    self.logger.info(f"Created unknown person: {person_name}")
                                else:
                                    self.logger.error(f"Failed to create unknown person: {person_name}")
                                    break
                            else:
                                # Add additional faces
                                if created_person_id:
                                    added = self.unknown_group.add_face_to_person(created_person_id, face_image)
                                    if added:
                                        self.logger.debug(f"Added face {i+1} to {person_name}")
                                    else:
                                        self.logger.warning(f"Failed to add face {i+1} to {person_name}")

                    success_count += 1

                except Exception as e:
                    self.logger.error(f"Error training person {person_data.get('name', 'unknown')}: {e}")

            # Trigger training
            if success_count > 0:
                if is_known:
                    self.known_group.train_group()
                else:
                    self.unknown_group.train_group()

                self.logger.info(f"Training completed: {success_count}/{total_count} persons processed")
                return True
            else:
                self.logger.warning("No persons were successfully processed for training")
                return False

        except Exception as e:
            self.logger.error(f"Error in train_from_structured_data: {e}")
            return False

    def get_file_system_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the file system structure

        Returns:
            dict: Statistics about saved data
        """
        try:
            stats = {
                'base_directory': str(self.base_save_directory),
                'save_processed_frames': self.save_processed_frames,
                'known_persons': 0,
                'unknown_persons': 0,
                'total_known_images': 0,
                'total_unknown_images': 0,
                'processed_frames': 0
            }

            # Count known persons and images
            known_persons = self.list_all_persons(True)
            stats['known_persons'] = len(known_persons)
            stats['total_known_images'] = sum(p.get('image_count', 0) for p in known_persons)

            # Count unknown persons and images
            unknown_persons = self.list_all_persons(False)
            stats['unknown_persons'] = len(unknown_persons)
            stats['total_unknown_images'] = sum(p.get('image_count', 0) for p in unknown_persons)

            # Count processed frames
            if self.save_processed_frames:
                frame_dir = self.base_save_directory / "frame_processed"
                if frame_dir.exists():
                    stats['processed_frames'] = len(list(frame_dir.glob("*.jpg")))

            return stats

        except Exception as e:
            self.logger.error(f"Error getting file system stats: {e}")
            return {}
    
    def get_training_status(self) -> Dict[str, str]:
        """Get training status of both groups"""
        return {
            'known': self.known_group.get_training_status(),
            'unknown': self.unknown_group.get_training_status()
        }
    
    def get_group_stats(self) -> Dict[str, Any]:
        """Get statistics for both groups"""
        return {
            'known': self.known_group.get_group_stats(),
            'unknown': self.unknown_group.get_group_stats()
        }

    def get_known_group_stats(self) -> Dict[str, Any]:
        """Get statistics for known group only"""
        return self.known_group.get_group_stats()

    def get_unknown_group_stats(self) -> Dict[str, Any]:
        """Get statistics for unknown group only"""
        return self.unknown_group.get_group_stats()
    
    def list_all_persons(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all persons from both groups"""
        return {
            'known': self.known_group.list_persons(),
            'unknown': self.unknown_group.list_persons()
        }
    
    def update_unknown_person(self, person_id: str, new_name: str, designation: str = None) -> bool:
        """Update an unknown person with real name and designation"""
        return self.unknown_group.update_unknown_person(person_id, new_name, designation)
    
    def delete_person(self, person_id: str, is_known: bool) -> bool:
        """Delete a person from the appropriate group"""
        if is_known:
            return self.known_group.delete_person(person_id)
        else:
            return self.unknown_group.delete_person(person_id)
    
    def identify_faces(self, face_ids: List[str], confidence_threshold: float = 0.6) -> List[DetectedFace]:
        """Identify faces against both groups"""
        try:
            # Identify against known group first
            known_results = self.known_group.identify_faces(face_ids, confidence_threshold)
            unknown_results = self.unknown_group.identify_faces(face_ids, confidence_threshold)
            
            detected_faces = []
            
            for face_id in face_ids:
                person_id = None
                person_name = "Unknown"
                confidence = 0.0
                is_known = False
                
                # Check known results first
                for result in known_results:
                    if result.face_id == face_id and result.candidates:
                        candidate = result.candidates[0]
                        if candidate.confidence >= confidence_threshold:
                            person_id = candidate.person_id
                            person_info = self.known_group.get_person_info(person_id)
                            person_name = person_info['name'] if person_info else "Unknown"
                            confidence = candidate.confidence
                            is_known = True
                            break
                
                # If not found in known, check unknown
                if not is_known:
                    for result in unknown_results:
                        if result.face_id == face_id and result.candidates:
                            candidate = result.candidates[0]
                            if candidate.confidence >= confidence_threshold:
                                person_id = candidate.person_id
                                person_info = self.unknown_group.get_person_info(person_id)
                                person_name = person_info['name'] if person_info else "Unknown"
                                confidence = candidate.confidence
                                is_known = False
                                break
                
                detected_face = DetectedFace(
                    face_id=face_id,
                    person_id=person_id,
                    person_name=person_name,
                    confidence=confidence,
                    is_known=is_known,
                    bounding_box={},  # Will be filled by caller
                    face_crop=None
                )
                
                detected_faces.append(detected_face)
            
            return detected_faces
            
        except Exception as e:
            self.logger.error(f"Error identifying faces: {e}")
            return []

    def cleanup_unknown_images(self):
        """Clean up all saved unknown images while keeping known images"""
        self.unknown_group.cleanup_unknown_images()

    def set_unknown_person_details(self, person_id: str, name: str, user_data: str = "") -> bool:
        """Set name and user_data for an unknown person"""
        success = self.unknown_group.set_person_details(person_id, name, user_data)

        if success:
            # Check if known group is currently training
            known_training_status = self.known_group.get_training_status()
            if known_training_status != "running":
                # Known group is not training, trigger training to process pending transfers
                self.logger.info("Triggering known group training due to new unknown person details")
                self._train_known_group_with_transfers()
            else:
                self.logger.info("Known group is currently training, pending transfer will be processed after training completes")

        return success

    def _train_known_group_with_transfers(self):
        """Train known group and process pending unknown transfers"""
        try:
            # First, process any pending unknown transfers
            transferred_count = self.known_group.process_pending_unknown_transfers(self.unknown_group)

            if transferred_count > 0:
                self.logger.info(f"Processed {transferred_count} pending transfers before training")

            # Train the known group
            self.known_group.train_group()

            # After training completes, check for any new pending transfers
            self._check_and_process_post_training_transfers()

        except Exception as e:
            self.logger.error(f"Error in train known group with transfers: {e}")

    def _check_and_process_post_training_transfers(self):
        """Check for new pending transfers after training and trigger another training if needed"""
        try:
            if self.unknown_group.has_pending_known_persons():
                self.logger.info("Found new pending transfers after training, scheduling another training cycle")
                # Schedule another training cycle for the new transfers
                self._train_known_group_with_transfers()

        except Exception as e:
            self.logger.error(f"Error checking post-training transfers: {e}")

    def train_all_groups(self):
        """Train all groups with intelligent transfer handling"""
        # Train known group with transfer processing
        self._train_known_group_with_transfers()

        # Train unknown group
        self.unknown_group.train_group()

    def load_faces_from_saved_images(self, group_type: str = "both") -> Dict[str, Any]:
        """
        Load face data from saved_images folder structure and add to groups for training

        Args:
            group_type: "known", "unknown", or "both" (default: "both")

        Returns:
            Dict with loading results and statistics
        """
        try:
            results = {
                'known': {'persons_loaded': 0, 'faces_loaded': 0, 'errors': []},
                'unknown': {'persons_loaded': 0, 'faces_loaded': 0, 'errors': []},
                'total_persons': 0,
                'total_faces': 0
            }

            # Load known persons if requested
            if group_type in ["known", "both"]:
                known_results = self._load_faces_from_group_folder("known")
                results['known'] = known_results
                results['total_persons'] += known_results['persons_loaded']
                results['total_faces'] += known_results['faces_loaded']

            # Load unknown persons if requested
            if group_type in ["unknown", "both"]:
                unknown_results = self._load_faces_from_group_folder("unknown")
                results['unknown'] = unknown_results
                results['total_persons'] += unknown_results['persons_loaded']
                results['total_faces'] += unknown_results['faces_loaded']

            self.logger.info(f"Loaded {results['total_persons']} persons with {results['total_faces']} faces from saved images")

            # Trigger training for loaded groups
            training_results = {'known_training': False, 'unknown_training': False}

            if group_type in ["known", "both"] and results['known']['persons_loaded'] > 0:
                self.logger.info("Starting training for known group after loading faces...")
                training_results['known_training'] = self.known_group.train_group()
                if training_results['known_training']:
                    self.logger.info("✅ Known group training completed after loading")
                else:
                    self.logger.warning("❌ Known group training failed after loading")

            if group_type in ["unknown", "both"] and results['unknown']['persons_loaded'] > 0:
                self.logger.info("Starting training for unknown group after loading faces...")
                training_results['unknown_training'] = self.unknown_group.train_group()
                if training_results['unknown_training']:
                    self.logger.info("✅ Unknown group training completed after loading")
                else:
                    self.logger.warning("❌ Unknown group training failed after loading")

            # Add training results to the response
            results['training'] = training_results

            return results

        except Exception as e:
            self.logger.error(f"Error loading faces from saved images: {e}")
            return {'error': str(e)}

    def _load_faces_from_group_folder(self, group_type: str) -> Dict[str, Any]:
        """Load faces from a specific group folder (known or unknown)"""
        import json
        import shutil

        results = {'persons_loaded': 0, 'faces_loaded': 0, 'errors': []}

        try:

            # Get the appropriate directory
            if group_type == "known":
                group_dir = Path("face_recognition/saved_images/known")
                target_group = self.known_group
            else:
                group_dir = Path("face_recognition/saved_images/unknown")
                target_group = self.unknown_group

            if not group_dir.exists():
                self.logger.warning(f"Group directory not found: {group_dir}")
                return results

            # Iterate through person directories
            for person_dir in group_dir.iterdir():
                if not person_dir.is_dir():
                    continue

                try:
                    # Load person data from JSON
                    json_file = person_dir / "person_data.json"
                    if not json_file.exists():
                        self.logger.warning(f"No person_data.json found in {person_dir}")
                        continue

                    with open(json_file, 'r', encoding='utf-8') as f:
                        person_data = json.load(f)

                    person_name = person_data.get('name', 'Unknown')
                    user_data = person_data.get('user_data', '')

                    # Create person in the group
                    if group_type == "known":
                        person_id = target_group.add_person(person_name, user_data)
                    else:
                        # For unknown group, we need to handle differently since it auto-generates names
                        person_id = None

                    if person_id or group_type == "unknown":
                        person_faces_loaded = 0
                        old_folder_path = person_dir

                        # Load all face images from the person directory
                        for image_file in person_dir.glob("*.jpg"):
                            if image_file.name == "person_data.json":
                                continue

                            try:
                                # Load image
                                face_image = cv2.imread(str(image_file))
                                if face_image is not None:
                                    # Add face to person (this will create new folder with new person_id)
                                    if group_type == "known" and person_id:
                                        success = target_group.add_face_to_person(person_id, face_image, user_data)
                                        if success:
                                            person_faces_loaded += 1
                                    elif group_type == "unknown":
                                        # For unknown group, add as new unknown person
                                        unknown_person_id = target_group.add_unknown_person(face_image, user_data)
                                        if unknown_person_id:
                                            person_faces_loaded += 1
                                            if not person_id:  # First face for this person
                                                person_id = unknown_person_id
                                else:
                                    self.logger.warning(f"Could not load image: {image_file}")

                            except Exception as e:
                                error_msg = f"Error loading face image {image_file}: {e}"
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)

                        if person_faces_loaded > 0:
                            results['persons_loaded'] += 1
                            results['faces_loaded'] += person_faces_loaded
                            self.logger.info(f"Loaded {person_name} with {person_faces_loaded} faces from {group_type} group")

                            # Clean up old folder to avoid redundancy and ensure folder name matches Azure person ID
                            try:
                                if old_folder_path.exists() and person_id:
                                    new_folder_path = group_dir / person_id

                                    # If the old folder name doesn't match the new person_id
                                    if old_folder_path.name != person_id:
                                        if new_folder_path.exists():
                                            # New folder already exists, remove old one
                                            self.logger.info(f"Removing old folder {old_folder_path.name} (replaced by {person_id})")
                                            shutil.rmtree(old_folder_path)
                                        else:
                                            # Rename old folder to match new person_id
                                            self.logger.info(f"Renaming folder {old_folder_path.name} to {person_id}")
                                            old_folder_path.rename(new_folder_path)

                                            # Update person_data.json with new person_id
                                            json_file = new_folder_path / "person_data.json"
                                            if json_file.exists():
                                                try:
                                                    with open(json_file, 'r') as f:
                                                        person_data = json.load(f)

                                                    # Update person_id in JSON
                                                    person_data['person_id'] = person_id
                                                    person_data['azure_person_id'] = person_id

                                                    with open(json_file, 'w') as f:
                                                        json.dump(person_data, f, indent=2)

                                                    self.logger.info(f"Updated person_data.json with new person_id: {person_id}")
                                                except Exception as e:
                                                    self.logger.warning(f"Could not update person_data.json: {e}")
                                    else:
                                        self.logger.debug(f"Folder name {old_folder_path.name} already matches person_id {person_id}")
                            except Exception as e:
                                self.logger.warning(f"Could not clean up old folder {old_folder_path}: {e}")
                        else:
                            self.logger.warning(f"No faces loaded for {person_name} in {group_type} group")
                    else:
                        error_msg = f"Failed to create person {person_name} in {group_type} group"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)

                except Exception as e:
                    error_msg = f"Error processing person directory {person_dir}: {e}"
                    self.logger.error(error_msg)
                    results['errors'].append(error_msg)

            return results

        except Exception as e:
            self.logger.error(f"Error loading faces from {group_type} folder: {e}")
            return {'persons_loaded': 0, 'faces_loaded': 0, 'errors': [str(e)]}

    def train_groups_from_saved_images(self, group_type: str = "both") -> bool:
        """
        Load faces from saved_images and train the groups

        Args:
            group_type: "known", "unknown", or "both" (default: "both")

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading faces from saved images and training {group_type} groups")

            # Load faces from saved images
            load_results = self.load_faces_from_saved_images(group_type)

            if 'error' in load_results:
                self.logger.error(f"Failed to load faces: {load_results['error']}")
                return False

            # Train the groups
            if group_type in ["known", "both"] and load_results['known']['persons_loaded'] > 0:
                self.logger.info(f"Training known group with {load_results['known']['persons_loaded']} persons")
                self.known_group.train_group()

            if group_type in ["unknown", "both"] and load_results['unknown']['persons_loaded'] > 0:
                self.logger.info(f"Training unknown group with {load_results['unknown']['persons_loaded']} persons")
                self.unknown_group.train_group()

            self.logger.info(f"Successfully loaded and trained {load_results['total_persons']} persons with {load_results['total_faces']} faces")
            return True

        except Exception as e:
            self.logger.error(f"Error training groups from saved images: {e}")
            return False
