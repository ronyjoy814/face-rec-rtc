# coding: utf-8

"""
Face Recognition Manager

High-level face recognition manager providing a comprehensive API for face
detection, identification, and person management. Built on modular architecture
with Azure AI Vision Face integration for enterprise-grade performance.

Key Features:
    - Complete face recognition pipeline
    - Person registration and management
    - Real-time face detection and identification
    - Training coordination and monitoring
    - Statistics and reporting
    - Professional API design
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .face_group_manager import FaceGroupManager, FrameProcessingResult, DetectedFace


class FaceRecognitionManager:
	"""
	Face Recognition Manager

	High-level interface for comprehensive face recognition operations.
	Provides enterprise-grade face detection, identification, and person
	management capabilities using Azure AI Vision Face API.
	"""
	
	def __init__(
		self,
		endpoint: str,
		api_key: str,
		group_id: str,
		group_name: str,
		auto_train_unknown: bool = True,
		max_faces_per_person: int = 10,
		unknown_confidence_threshold: float = 0.8,
		face_similarity_threshold: float = 0.8,
		save_known_images: bool = False,
		cleanup_known_group=False,
		cleanup_unknown_images=True,
		logger: Optional[logging.Logger] = None
	):
		"""
		Initialize Face Recognition Manager

		Args:
			endpoint: Azure Face API endpoint
			api_key: Azure Face API key
			group_id: Base group ID for all groups
			group_name: Base group name
			auto_train_unknown: Enable automatic training for unknown faces
			max_faces_per_person: Maximum faces per person (enables auto-addition)
			unknown_confidence_threshold: Confidence threshold for unknown faces
			face_similarity_threshold: Similarity threshold for face matching
			save_known_images: Whether to save known face images during training
			logger: Optional logger instance
		"""
		self.endpoint = endpoint
		self.api_key = api_key
		self.group_id = group_id
		self.group_name = group_name
		self.auto_train_unknown = auto_train_unknown
		self.max_faces_per_person = max_faces_per_person
		self.unknown_confidence_threshold = unknown_confidence_threshold
		self.face_similarity_threshold = face_similarity_threshold
		self.save_known_images = save_known_images
		self.cleanup_known_group = cleanup_known_group
		self.cleanup_unknown_images_flag = cleanup_unknown_images
		
		# Setup logger
		self.logger = logger or self._setup_logger()
		
		# Initialize the group manager (does all the heavy lifting)
		self.group_manager = FaceGroupManager(
			endpoint=endpoint,
			api_key=api_key,
			group_id=group_id,
			group_name=group_name,
			auto_train_unknown=auto_train_unknown,
			max_faces_per_person=max_faces_per_person,
			unknown_confidence_threshold=unknown_confidence_threshold,
			face_similarity_threshold=face_similarity_threshold,
			save_known_images=save_known_images,
			save_unknown_images=True,
			cleanup_unknown_images=cleanup_unknown_images,
			cleanup_known_group=cleanup_known_group,
			logger=logger
		)
		
		self.logger.info(f"Initialized FaceRecognitionManager: {group_name}")
	
	def _setup_logger(self) -> logging.Logger:
		"""Setup logger for the manager"""
		logger = logging.getLogger(f"{self.__class__.__name__}")
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter(
				'%(asctime)s %(levelname)s %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S'
			)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
			logger.setLevel(logging.DEBUG)
		return logger

	def get_known_group_id(self) -> str:
		"""Get the known group ID"""
		return self.group_manager.get_known_group_id()

	def get_unknown_group_id(self) -> str:
		"""Get the unknown group ID"""
		return self.group_manager.get_unknown_group_id()

	# Main frame processing method
	def process_frame(
		self,
		frame: np.ndarray,
		confidence_threshold: float = 0.6,
		return_known_crops: bool = True,
		auto_train_unknown: bool = None,
		auto_add_faces: bool = True,
		similarity_threshold: float = 0.8,
		time_gap_seconds: int = 5,
		max_faces_per_person: int = 5
	) -> FrameProcessingResult:
		"""
		Process video frame and identify faces with auto-addition capabilities

		Args:
			frame: Input video frame
			confidence_threshold: Minimum confidence for face identification
			return_known_crops: Whether to return cropped images for known faces
			auto_train_unknown: Override auto-training setting for this frame
			auto_add_faces: Whether to automatically add faces based on similarity
			similarity_threshold: Minimum similarity for face matching
			time_gap_seconds: Minimum time gap between faces (default: 5 seconds)
			max_faces_per_person: Maximum faces per person (default: 5)

		Returns:
			FrameProcessingResult with detected faces and statistics
		"""
		return self.group_manager.process_frame(
			frame=frame,
			auto_add_faces=auto_add_faces,
			similarity_threshold=similarity_threshold,
			confidence_threshold=confidence_threshold,
			time_gap_seconds=time_gap_seconds,
			max_faces_per_person=max_faces_per_person
		)

	def process_frame_optimized(
		self,
		frame: np.ndarray,
		confidence_threshold: float = 0.6,
		return_known_crops: bool = True,
		auto_train_unknown: bool = None,
		auto_add_faces: bool = True,
		similarity_threshold: float = 0.8,
		time_gap_seconds: int = 5,
		max_faces_per_person: int = 5
	) -> FrameProcessingResult:
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
			frame: Input video frame
			confidence_threshold: Minimum confidence for face identification
			return_known_crops: Whether to return cropped images for known faces
			auto_train_unknown: Override auto-training setting for this frame
			auto_add_faces: Whether to automatically add faces based on similarity
			similarity_threshold: Minimum similarity for face matching
			time_gap_seconds: Minimum time gap between faces (default: 5 seconds)
			max_faces_per_person: Maximum faces per person (default: 5)

		Returns:
			FrameProcessingResult with immediate processing results and tagged faces
		"""
		return self.group_manager.process_frame(
			frame=frame,
			auto_add_faces=auto_add_faces,
			similarity_threshold=similarity_threshold,
			confidence_threshold=confidence_threshold,
			time_gap_seconds=time_gap_seconds,
			max_faces_per_person=max_faces_per_person
		)

	# Person mapping and transfer functionality
	def map_unknown_person_to_known(self, name: str, user_data: str = "",
	                                identifier = None, similarity_threshold: float = 0.9,
	                                confidence_threshold: float = 0.9) -> bool:
		"""
		Map unknown person to known identity using queue-based communication

		Args:
			name: Name for the known person
			user_data: User data for the known person
			identifier: Either unknown person name (e.g., "person_1") OR an image for similarity matching
			similarity_threshold: Minimum similarity for image-based identification (default: 0.9)
			confidence_threshold: Minimum confidence for image-based identification (default: 0.9)

		Returns:
			bool: True if mapping was successfully queued, False otherwise
		"""
		try:
			unknown_person_id = None

			if isinstance(identifier, str):
				# Direct mapping by person name (e.g., "person_1")
				unknown_person_id = self.group_manager.unknown_group.get_person_id_by_name(identifier)
				if not unknown_person_id:
					self.logger.error(f"Unknown person with name '{identifier}' not found")
					return False

			elif identifier is not None:
				# Image-based similarity matching
				import numpy as np
				if isinstance(identifier, np.ndarray):
					# Find matching unknown person using strict similarity
					unknown_results = self.group_manager.unknown_group.identify_faces([identifier])

					if unknown_results and len(unknown_results) > 0:
						best_match = max(unknown_results, key=lambda x: x.get('confidence', 0))
						confidence = best_match.get('confidence', 0)
						similarity = best_match.get('similarity', 0)

						if confidence >= confidence_threshold and similarity >= similarity_threshold:
							unknown_person_id = best_match.get('person_id')
							self.logger.info(f"Found matching unknown person with confidence: {confidence:.3f}, similarity: {similarity:.3f}")
						else:
							self.logger.error(f"No unknown person found with sufficient similarity ({similarity:.3f} < {similarity_threshold}) and confidence ({confidence:.3f} < {confidence_threshold})")
							return False
					else:
						self.logger.error("No matching unknown person found for provided image")
						return False
				else:
					self.logger.error("Invalid identifier type. Must be string (person name) or numpy array (image)")
					return False
			else:
				self.logger.error("Identifier must be provided (person name or image)")
				return False

			if unknown_person_id:
				# Queue the transfer using the queue-based communication system
				success = self.group_manager.queue_person_transfer(
					unknown_person_id=unknown_person_id,
					known_person_name=name,
					known_user_data=user_data
				)

				if success:
					self.logger.info(f"Successfully queued transfer of unknown person to known as '{name}'")
					return True
				else:
					self.logger.error("Failed to queue person transfer")
					return False
			else:
				self.logger.error("Could not determine unknown person ID")
				return False

		except Exception as e:
			self.logger.error(f"Error mapping unknown person to known: {e}")
			return False

	# Known person management
	def create_known_person(self, name: str, user_data: str = "") -> str | None:
		"""
		Create a new known person

		Args:
			name: Person's name
			user_data: Person's designation or short summary (e.g., "Manager", "Developer")

		Returns:
			Person ID if successful, None otherwise
		"""
		return self.group_manager.create_known_person(name, user_data)
	
	def add_face_to_known_person(self, person_id: str, face_image: np.ndarray, user_data: str = "") -> str | None:
		"""
		Add face image to existing known person

		Args:
			person_id: ID of the person
			face_image: Face image (numpy array) to add to the person
			user_data: Additional context (e.g., "Photo from meeting", "ID card photo")

		Returns:
			Face ID if successful, None otherwise
		"""
		return self.group_manager.add_face_to_known_person(person_id, face_image, user_data)
	
	def get_known_persons(self) -> List[Any]:
		"""
		Get all known persons from both known groups
		
		Returns:
			List of person objects with group information
		"""
		return self.group_manager.get_known_persons()
	
	# Unknown person management
	def update_unknown_person_with_real_name(self, person_label: str, real_name: str, user_data: str = "") -> bool:
		"""
		Update an unknown person (person_X) with real name and details

		Args:
			person_label: Current auto-generated label (e.g., "person_1", "person_2")
			real_name: Real name to update to (e.g., "John Smith")
			user_data: Person's designation or short summary (e.g., "Manager", "Visitor", "Security Guard")

		Returns:
			True if successful, False otherwise
		"""
		return self.group_manager.update_unknown_person_with_real_name(person_label, real_name, user_data)
	
	def get_unknown_persons_list(self) -> list[dict]:
		"""
		Get list of all unknown persons with metadata
		
		Returns:
			List of dictionaries with person information
		"""
		return self.group_manager.get_unknown_persons_list()
	
	# Statistics and monitoring
	def get_group_statistics(self) -> Dict[str, Any]:
		"""
		Get comprehensive statistics for all groups
		
		Returns:
			Dictionary with statistics for all 4 groups
		"""
		return self.group_manager.get_group_stats()
	
	def get_training_status(self) -> Dict[str, Any]:
		"""
		Get training status for all groups
		
		Returns:
			Dictionary with training status for all groups
		"""
		return self.group_manager.get_training_status()

	def get_group_stats(self) -> Dict[str, Any]:
		"""Get statistics for both groups"""
		return self.group_manager.get_group_stats()

	def list_all_persons(self) -> Dict[str, List[Dict[str, Any]]]:
		"""List all persons from both groups"""
		return self.group_manager.list_all_persons()

	def add_known_person(self, person_name: str, face_image: np.ndarray) -> Optional[str]:
		"""Add a known person with face image"""
		return self.group_manager.add_known_person(person_name, face_image)

	def train_all_groups(self):
		"""Train both known and unknown groups"""
		return self.group_manager.train_all_groups()

	def get_known_group_stats(self) -> Dict[str, Any]:
		"""Get statistics for known group"""
		return self.group_manager.get_known_group_stats()

	def get_unknown_group_stats(self) -> Dict[str, Any]:
		"""Get statistics for unknown group"""
		return self.group_manager.get_unknown_group_stats()

	# Utility methods
	def save_cropped_faces(self, detected_faces: List[DetectedFace], output_dir: str = "cropped_faces") -> List[str]:
		"""
		Save cropped face images to directory
		
		Args:
			detected_faces: List of detected faces with cropped images
			output_dir: Output directory for saved images
			
		Returns:
			List of saved file paths
		"""
		import os
		import cv2
		from datetime import datetime
		
		os.makedirs(output_dir, exist_ok=True)
		saved_files = []
		
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		
		for i, face in enumerate(detected_faces):
			if face.cropped_image is not None:
				# Create filename
				safe_name = face.tag_text.replace(" ", "_").replace("/", "_")
				filename = f"{timestamp}_{i:02d}_{safe_name}_{face.confidence:.2f}.jpg"
				filepath = os.path.join(output_dir, filename)
				
				# Save image
				try:
					cv2.imwrite(filepath, face.cropped_image)
					saved_files.append(filepath)
					self.logger.info(f"Saved cropped face: {filepath}")
				except Exception as e:
					self.logger.error(f"Error saving cropped face {filepath}: {e}")
		
		return saved_files
	
	def __str__(self) -> str:
		"""String representation of the manager"""
		return f"FaceRecognitionManager(group_id='{self.group_id}', name='{self.group_name}')"
	
	def __repr__(self) -> str:
		"""Detailed representation of the manager"""
		return (f"FaceRecognitionManager(group_id='{self.group_id}', "
				f"auto_train={self.auto_train_unknown}, "
				f"max_faces={self.max_faces_per_person})")

	def cleanup_unknown_images(self):
		"""Clean up all saved unknown images while keeping known images"""
		self.group_manager.cleanup_unknown_images()

	def set_unknown_person_details(self, person_id: str, name: str, user_data: str = "") -> bool:
		"""Set name and user_data for an unknown person, triggering transfer to known group"""
		return self.group_manager.set_unknown_person_details(person_id, name, user_data)

	def get_pending_unknown_persons(self) -> Dict[str, Dict[str, Any]]:
		"""Get all unknown persons that have details set and are pending transfer"""
		return self.group_manager.unknown_group.get_pending_known_persons()
