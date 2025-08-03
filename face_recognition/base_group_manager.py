# coding: utf-8

"""
Base Group Manager

Foundation class for face group management providing common operations
and utilities for Azure AI Vision Face API integration. Serves as the
base for both known and unknown face group implementations.

Key Features:
    - Azure Face API client management
    - Group lifecycle operations
    - Training coordination
    - Image processing utilities
    - Error handling and logging
    - Performance monitoring
"""

import logging
import numpy as np
import cv2
import threading
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from azure.ai.vision.face import FaceClient, FaceAdministrationClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from azure.core.credentials import AzureKeyCredential
import base64
from io import BytesIO
from pathlib import Path

class BaseGroupManager:
	"""
	Base Group Manager

	Foundation class providing common functionality for face group management.
	Handles Azure AI Vision Face API interactions, group operations, training
	coordination, and image processing utilities.
	"""
	
	def __init__(
		self,
		endpoint: str,
		api_key: str,
		max_faces_per_person: int = 10,
		face_similarity_threshold: float = 0.8,
		logger: Optional[logging.Logger] = None
	):
		"""Initialize base group manager"""
		self.endpoint = endpoint
		self.api_key = api_key
		self.max_faces_per_person = max_faces_per_person
		self.face_similarity_threshold = face_similarity_threshold
		
		# Setup logger
		self.logger = logger or self._setup_logger()
		
		# Initialize Azure Face API clients
		credential = AzureKeyCredential(api_key)
		self.face_client = FaceClient(endpoint, credential)
		self.face_admin_client = FaceAdministrationClient(endpoint, credential)

		# Simple group ID for single group approach
		self.group_id = None

		# Image saving configuration
		self.face_images_dir = Path("face_recognition/saved_images")
		self.known_images_dir = self.face_images_dir / "known"
		self.unknown_images_dir = self.face_images_dir / "unknown"

		# Create directories
		self.known_images_dir.mkdir(parents=True, exist_ok=True)
		self.unknown_images_dir.mkdir(parents=True, exist_ok=True)
		
	def _setup_logger(self) -> logging.Logger:
		"""Setup logger for the group manager"""
		logger = logging.getLogger(f"{self.__class__.__name__}")
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter(
				'%(asctime)s %(levelname)s %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S'
			)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
			logger.setLevel(logging.INFO)
		return logger
	
	def create_group_if_not_exists(self, group_id: str, group_name: str) -> bool:
		"""Create large person group if it doesn't exist"""
		try:
			# Check if group exists
			try:
				self.face_admin_client.large_person_group.get(group_id)
				self.logger.info(f"Group {group_id} already exists")
				return True
			except Exception:
				# Group doesn't exist, create it
				self.face_admin_client.large_person_group.create(
					group_id,
					name=group_name,
					recognition_model=FaceRecognitionModel.RECOGNITION04
				)
				self.logger.info(f"Created group: {group_id}")
				return True
				
		except Exception as e:
			self.logger.error(f"Error creating group {group_id}: {e}")
			return False
	
	def create_person_in_group(self, group_id: str, name: str, user_data: str = "") -> str | None:
		"""Create a person in specified group with timestamp tracking"""
		try:
			# Add timestamp to user_data for tracking
			timestamp = datetime.now().isoformat()
			timestamped_user_data = f"{user_data} | Created: {timestamp}"
			
			person = self.face_admin_client.large_person_group.create_person(
				group_id, 
				name=name, 
				user_data=timestamped_user_data
			)
			self.logger.info(f"Created person {name} with ID: {person.person_id} in group {group_id} at {timestamp}")
			return person.person_id
		except Exception as e:
			self.logger.error(f"Error creating person in group {group_id}: {e}")
			return None
	
	def add_face_to_person_in_group(self, group_id: str, person_id: str,
									face_image: np.ndarray, user_data: str = "") -> str | None:
		"""Add face to person in specified group with timestamp tracking"""
		try:
			image_bytes = self._image_to_bytes(face_image)
			
			# Add timestamp to user_data for tracking
			timestamp = datetime.now().isoformat()
			timestamped_user_data = f"{user_data} | Added: {timestamp}"
			
			face = self.face_admin_client.large_person_group.add_face(
				large_person_group_id=group_id,
				person_id=person_id,
				image_content=image_bytes,
				user_data=timestamped_user_data,
				detection_model=FaceDetectionModel.DETECTION03
			)
			self.logger.info(f"Added face {face.persisted_face_id} to person {person_id} in group {group_id} at {timestamp}")
			return face.persisted_face_id
		except Exception as e:
			self.logger.error(f"Error adding face to person {person_id} in group {group_id}: {e}")
			return None
	
	def delete_person_from_group(self, group_id: str, person_id: str) -> bool:
		"""Delete person from specified group"""
		try:
			self.face_admin_client.large_person_group.delete_person(group_id, person_id)
			self.logger.info(f"Successfully deleted person: {person_id}")
			return True
		except Exception as e:
			self.logger.error(f"Error deleting person {person_id}: {e}")
			return False
	
	def get_persons_in_group(self, group_id: str) -> List[Any]:
		"""Get all persons in specified group"""
		try:
			return self.face_admin_client.large_person_group.get_persons(group_id)
		except Exception as e:
			self.logger.error(f"Error getting persons from group {group_id}: {e}")
			return []
	
	def trigger_training(self):
		"""Trigger training for the group"""
		try:
			self.face_admin_client.large_person_group.begin_train(self.group_id)
			self.logger.info(f"Training started for group: {self.group_id}")
		except Exception as e:
			self.logger.error(f"Error starting training for group {self.group_id}: {e}")

	
	
	def train_group_async(self, group_id: str, group_type: str):
		"""Start asynchronous training for specified group (legacy method)"""
		self.add_to_training_queue(group_id, group_type)

	def save_unknown_face_image(self, face_image: np.ndarray, person_id: str, face_id: str) -> str | None:
		"""Save unknown face image to disk (always save for unknown faces)"""
		try:
			# Create person directory: saved_images/unknown/person_id/
			person_dir = self.unknown_images_dir / person_id
			person_dir.mkdir(parents=True, exist_ok=True)

			# Check if we already have max images for this person
			existing_images = list(person_dir.glob("*.jpg"))
			if len(existing_images) >= self.max_faces_per_person:
				# Remove oldest image to make space
				oldest_image = min(existing_images, key=lambda f: f.stat().st_mtime)
				oldest_image.unlink()
				self.logger.info(f"Removed oldest image to make space: {oldest_image}")

			# Generate filename with timestamp
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			filename = f"{face_id}_{timestamp}.jpg"
			file_path = person_dir / filename

			# Save image
			cv2.imwrite(str(file_path), face_image)

			self.logger.info(f"Saved unknown face image: {file_path}")
			return str(file_path)

		except Exception as e:
			self.logger.error(f"Error saving unknown face image: {e}")
			return None

	def save_known_face_image(self, face_image: np.ndarray, person_id: str, face_id: str,
							  person_name: str, user_data: str = "") -> str | None:
		"""Save known face image to disk (only when training)"""
		try:
			# Create person directory: saved_images/known/person_id/
			person_dir = self.known_images_dir / person_id
			person_dir.mkdir(parents=True, exist_ok=True)

			# Save person info to JSON file
			person_info = {
				"person_id": person_id,
				"name": person_name,
				"user_data": user_data,
				"created_at": datetime.now().isoformat(),
				"last_updated": datetime.now().isoformat()
			}

			json_file = person_dir / "person_info.json"
			with open(json_file, 'w') as f:
				json.dump(person_info, f, indent=2)

			# Generate filename with timestamp
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			filename = f"{face_id}_{timestamp}.jpg"
			file_path = person_dir / filename

			# Save image
			cv2.imwrite(str(file_path), face_image)

			self.logger.info(f"Saved known face image: {file_path}")
			return str(file_path)

		except Exception as e:
			self.logger.error(f"Error saving known face image: {e}")
			return None

	def get_unknown_person_images(self, person_id: str) -> List[str]:
		"""Get list of saved image paths for an unknown person"""
		try:
			person_dir = self.unknown_images_dir / person_id
			if not person_dir.exists():
				return []

			# Get all jpg files in person directory
			image_files = list(person_dir.glob("*.jpg"))
			return [str(f) for f in image_files]

		except Exception as e:
			self.logger.error(f"Error getting unknown person images: {e}")
			return []

	def get_known_person_info(self, person_id: str) -> Dict[str, Any] | None:
		"""Get known person info from JSON file"""
		try:
			person_dir = self.known_images_dir / person_id
			json_file = person_dir / "person_info.json"

			if not json_file.exists():
				return None

			with open(json_file, 'r') as f:
				return json.load(f)

		except Exception as e:
			self.logger.error(f"Error getting known person info: {e}")
			return None

	def transfer_unknown_to_known(self, unknown_person_id: str, known_person_id: str,
								  person_name: str, user_data: str = "") -> bool:
		"""Transfer unknown person images to known person folder"""
		try:
			unknown_dir = self.unknown_images_dir / unknown_person_id
			known_dir = self.known_images_dir / known_person_id

			if not unknown_dir.exists():
				self.logger.warning(f"No unknown person directory found: {unknown_person_id}")
				return False

			# Create known person directory
			known_dir.mkdir(parents=True, exist_ok=True)

			# Save person info to JSON file
			person_info = {
				"person_id": known_person_id,
				"name": person_name,
				"user_data": user_data,
				"created_at": datetime.now().isoformat(),
				"last_updated": datetime.now().isoformat(),
				"transferred_from_unknown": unknown_person_id
			}

			json_file = known_dir / "person_info.json"
			with open(json_file, 'w') as f:
				json.dump(person_info, f, indent=2)

			# Move all image files
			moved_count = 0
			for image_file in unknown_dir.glob("*.jpg"):
				new_path = known_dir / image_file.name
				image_file.rename(new_path)
				moved_count += 1

			# Remove empty unknown directory
			if moved_count > 0:
				unknown_dir.rmdir()

			self.logger.info(f"Transferred {moved_count} images from unknown person {unknown_person_id} to known person {known_person_id}")
			return True

		except Exception as e:
			self.logger.error(f"Error transferring unknown to known: {e}")
			return False

	def save_face_image(self, face_image: np.ndarray, person_id: str, face_id: str,
						group_type: str = "unknown", save_enabled: bool = True) -> str | None:
		"""Save face image to disk and return file path"""
		if not save_enabled:
			return None

		try:
			# Create directory structure: saved_images/group_type/person_id/
			person_dir = self.face_images_dir / group_type / person_id
			person_dir.mkdir(parents=True, exist_ok=True)

			# Generate filename with timestamp
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			filename = f"{face_id}_{timestamp}.jpg"
			file_path = person_dir / filename

			# Save image
			cv2.imwrite(str(file_path), face_image)

			self.logger.info(f"Saved face image: {file_path}")
			return str(file_path)

		except Exception as e:
			self.logger.error(f"Error saving face image: {e}")
			return None

	def get_saved_face_images(self, person_id: str, group_type: str = "unknown") -> List[str]:
		"""Get list of saved face image paths for a person"""
		try:
			person_dir = self.face_images_dir / group_type / person_id
			if not person_dir.exists():
				return []

			# Get all jpg files in person directory
			image_files = list(person_dir.glob("*.jpg"))
			return [str(f) for f in image_files]

		except Exception as e:
			self.logger.error(f"Error getting saved face images: {e}")
			return []

	def move_person_images(self, person_id: str, from_group_type: str, to_group_type: str) -> bool:
		"""Move all saved images for a person from one group type to another"""
		try:
			from_dir = self.face_images_dir / from_group_type / person_id
			to_dir = self.face_images_dir / to_group_type / person_id

			if not from_dir.exists():
				self.logger.warning(f"No saved images found for person {person_id} in {from_group_type}")
				return True  # Not an error if no images exist

			# Create destination directory
			to_dir.mkdir(parents=True, exist_ok=True)

			# Move all image files
			moved_count = 0
			for image_file in from_dir.glob("*.jpg"):
				new_path = to_dir / image_file.name
				image_file.rename(new_path)
				moved_count += 1

			# Remove empty source directory
			if moved_count > 0:
				from_dir.rmdir()

			self.logger.info(f"Moved {moved_count} images for person {person_id} from {from_group_type} to {to_group_type}")
			return True

		except Exception as e:
			self.logger.error(f"Error moving person images: {e}")
			return False
	
	def get_training_status(self, group_id: str) -> Dict[str, Any]:
		"""Get training status for specified group"""
		try:
			training_status = self.face_admin_client.large_person_group.get_training_status(group_id)
			return {
				'training': self.training_states.get(group_id, False),
				'status': training_status.status.value if training_status.status else 'unknown',
				'created': training_status.created_date_time.isoformat() if training_status.created_date_time else None,
				'last_action': training_status.last_action_date_time.isoformat() if training_status.last_action_date_time else None,
				'message': training_status.message or ''
			}
		except Exception as e:
			self.logger.warning(f"Could not get training status for {group_id}: {e}")
			return {
				'training': self.training_states.get(group_id, False),
				'status': 'unknown',
				'created': None,
				'last_action': None,
				'message': str(e)
			}
	
	def _image_to_bytes(self, image: np.ndarray) -> bytes:
		"""Convert numpy image to bytes"""
		try:
			# Convert BGR to RGB if needed
			if len(image.shape) == 3 and image.shape[2] == 3:
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			else:
				image_rgb = image
			
			# Encode as JPEG
			from PIL import Image
			pil_image = Image.fromarray(image_rgb)
			
			# Save to bytes
			buffer = BytesIO()
			pil_image.save(buffer, format='JPEG', quality=95)
			return buffer.getvalue()
			
		except Exception as e:
			self.logger.error(f"Error converting image to bytes: {e}")
			raise
	
	def _crop_face_from_frame(self, frame: np.ndarray, detected_face) -> np.ndarray | None:
		"""Crop face from frame using detection rectangle"""
		try:
			if not detected_face or not detected_face.face_rectangle:
				return None
			
			rect = detected_face.face_rectangle
			height, width = frame.shape[:2]
			
			# Calculate crop coordinates with padding
			padding = 20
			left = max(0, rect.left - padding)
			top = max(0, rect.top - padding)
			right = min(width, rect.left + rect.width + padding)
			bottom = min(height, rect.top + rect.height + padding)
			
			# Crop face
			face_crop = frame[top:bottom, left:right]
			
			if face_crop.size == 0:
				return None
			
			return face_crop
			
		except Exception as e:
			self.logger.error(f"Error cropping face: {e}")
			return None
	
	def _assess_face_quality(self, detected_face) -> float:
		"""Assess face quality based on detection attributes"""
		try:
			quality_score = 0.5  # Base score
			
			if detected_face.face_attributes:
				# Add quality factors if available
				if hasattr(detected_face.face_attributes, 'blur') and detected_face.face_attributes.blur:
					blur_level = detected_face.face_attributes.blur.blur_level
					if blur_level == 'low':
						quality_score += 0.3
					elif blur_level == 'medium':
						quality_score += 0.1
				
				if hasattr(detected_face.face_attributes, 'exposure') and detected_face.face_attributes.exposure:
					exposure_level = detected_face.face_attributes.exposure.exposure_level
					if exposure_level == 'goodExposure':
						quality_score += 0.2
			
			return min(1.0, quality_score)
			
		except Exception as e:
			self.logger.warning(f"Error assessing face quality: {e}")
			return 0.5
	
	def cleanup_group(self, group_id: str=None) -> bool:
		"""Delete all persons from group"""
		try:
			if group_id is None:
				group_id = self.group_id
			persons = self.get_persons_in_group(group_id)
			
			for person in persons:
				self.delete_person_from_group(group_id, person.person_id)
			
			self.logger.info(f"Cleaned up group {group_id}: deleted {len(persons)} persons")
			return True
			
		except Exception as e:
			self.logger.error(f"Error cleaning up group {group_id}: {e}")
			return False

def create_group_if_not_exists(self, group_id: str, group_name: str):
	"""Create group if it doesn't exist"""
	try:
		# Check if group exists
		try:
			self.face_admin_client.large_person_group.get(group_id)
			self.logger.info(f"Group {group_id} already exists")
		except Exception:
			# Group doesn't exist, create it
			from azure.ai.vision.face.models import FaceRecognitionModel
			self.face_admin_client.large_person_group.create(
				group_id,
				name=group_name,
				recognition_model=FaceRecognitionModel.RECOGNITION04
			)
			self.logger.info(f"Created group: {group_id}")

	except Exception as e:
		self.logger.error(f"Error creating group {group_id}: {e}")
		raise

def image_to_bytes(self, image):
	"""Convert OpenCV image to bytes"""
	try:
		import cv2
		success, encoded_image = cv2.imencode('.jpg', image)
		if success:
			return encoded_image.tobytes()
		return None
	except Exception as e:
		self.logger.error(f"Error converting image to bytes: {e}")
		return None
