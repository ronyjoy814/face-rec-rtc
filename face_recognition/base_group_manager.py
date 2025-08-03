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
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from azure.ai.vision.face import FaceClient, FaceAdministrationClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceTrainingResult
from azure.core.credentials import AzureKeyCredential
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
		logger: Optional[logging.Logger] = None,
		log_level: int = logging.INFO
	):
		"""Initialize base group manager"""
		self.endpoint = endpoint
		self.api_key = api_key
		self.max_faces_per_person = max_faces_per_person
		self.face_similarity_threshold = face_similarity_threshold
		self.log_level = log_level

		# Setup logger
		self.logger = logger or self._setup_logger(log_level)
		
		# Initialize Azure Face API clients
		credential = AzureKeyCredential(api_key)
		self.face_client = FaceClient(endpoint, credential)
		self.face_admin_client = FaceAdministrationClient(endpoint, credential)

		# Simple group ID for single group approach
		self.group_id = None

		# Training states tracking
		self.training_states = {}  # Track training states for different groups

		# Image saving configuration
		self.face_images_dir = Path("face_recognition/saved_images")
		self.known_images_dir = self.face_images_dir / "known"
		self.unknown_images_dir = self.face_images_dir / "unknown"

		# Create directories
		self.known_images_dir.mkdir(parents=True, exist_ok=True)
		self.unknown_images_dir.mkdir(parents=True, exist_ok=True)
		
	def _setup_logger(self, log_level: int = logging.INFO) -> logging.Logger:
		"""Setup logger for the group manager"""
		logger = logging.getLogger(f"{self.__class__.__name__}")
		if not logger.handlers:
			handler = logging.StreamHandler()
			formatter = logging.Formatter(
				'%(asctime)s %(name)s %(levelname)s %(message)s',
				datefmt='%Y-%m-%d %H:%M:%S'
			)
			handler.setFormatter(formatter)
			logger.addHandler(handler)
			logger.setLevel(log_level)
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
	

	
	def get_persons_in_group(self, group_id: str) -> List[Any]:
		"""Get all persons in specified group"""
		try:
			return self.face_admin_client.large_person_group.get_persons(group_id)
		except Exception as e:
			self.logger.error(f"Error getting persons from group {group_id}: {e}")
			return []

	def list_persons_in_group(self, group_id: str) -> List[Dict[str, Any]]:
		"""List all persons in specified group with formatted data"""
		try:
			persons = self.face_admin_client.large_person_group.get_persons(group_id)
			return [
				{
					'person_id': person.person_id,
					'name': person.name,
					'user_data': person.user_data,
					'face_count': len(person.persisted_face_ids) if person.persisted_face_ids else 0
				}
				for person in persons
			]

		except Exception as e:
			self.logger.error(f"Error listing persons in group {group_id}: {e}")
			return []

	def delete_person_from_group(self, group_id: str, person_id: str) -> bool:
		"""Delete a person from specified group"""
		try:
			self.face_admin_client.large_person_group.delete_person(group_id, person_id)
			self.logger.info(f"Successfully deleted person {person_id} from group {group_id}")
			return True
		except Exception as e:
			self.logger.error(f"Error deleting person {person_id} from group {group_id}: {e}")
			return False

	def delete_person_with_training(self, group_id: str, person_id: str, group_type: str = "group") -> bool:
		"""Delete a person from specified group and trigger training"""
		try:
			self.face_admin_client.large_person_group.delete_person(group_id, person_id)
			self.logger.info(f"Successfully deleted person {person_id} from {group_type}")

			# Trigger training after person deletion
			self.logger.info(f"Starting {group_type} training after person deletion")
			training_success = self.train_group()
			if training_success:
				self.logger.info(f"✅ {group_type} training completed after deletion")
			else:
				self.logger.warning(f"❌ {group_type} training failed after deletion")

			return True

		except Exception as e:
			self.logger.error(f"Error deleting person {person_id} from {group_type}: {e}")
			return False

	def get_person_name_from_api(self, person_id: str, group_id: str) -> str | None:
		"""Get person name by person ID from API - consolidated method"""
		try:
			person = self.face_admin_client.large_person_group.get_person(
				large_person_group_id=group_id,
				person_id=person_id
			)
			return person.name if person else None
		except Exception as e:
			self.logger.error(f"Error getting person name for {person_id}: {e}")
			return None

	def set_group_manager_reference(self, group_manager):
		"""Set reference to the parent group manager for file operations - consolidated method"""
		self.group_manager = group_manager

	def get_person_faces_from_api(self, person_id: str, group_id: str) -> List[Dict[str, Any]]:
		"""Get all faces for a specific person from API - consolidated method"""
		try:
			# Get person info which contains persisted_face_ids
			person = self.face_admin_client.large_person_group.get_person(
				large_person_group_id=group_id,
				person_id=person_id
			)

			face_list = []
			if person and person.persisted_face_ids:
				for face_id in person.persisted_face_ids:
					try:
						# Get individual face details
						face = self.face_admin_client.large_person_group.get_face(
							large_person_group_id=group_id,
							person_id=person_id,
							persisted_face_id=face_id
						)
						face_info = {
							'face_id': face.persisted_face_id,
							'user_data': face.user_data or ''
						}
						face_list.append(face_info)
					except Exception as face_error:
						self.logger.debug(f"Could not get face {face_id}: {face_error}")

			return face_list

		except Exception as e:
			self.logger.error(f"Error getting person faces from group {group_id}: {e}")
			return []
	
	def train_group(self, group_id: str = None) -> bool:
		"""Train the specified group (or default group if none specified)"""
		try:
			# Use provided group_id or fall back to instance group_id
			target_group_id = group_id or self.group_id

			# Check if training is already running
			current_status = self.get_training_status_simple(target_group_id)

			if current_status == "running":
				self.logger.info(f"Training already running for group: {target_group_id}, waiting for completion...")

				# Wait for current training to complete
				import time
				max_wait_time = 300  # 5 minutes max wait
				wait_interval = 5    # Check every 5 seconds
				waited_time = 0

				while current_status == "running" and waited_time < max_wait_time:
					time.sleep(wait_interval)
					waited_time += wait_interval
					current_status = self.get_training_status_simple(target_group_id)
					self.logger.debug(f"Waiting for training completion... Status: {current_status} ({waited_time}s)")

				if current_status == "running":
					self.logger.warning(f"Training still running after {max_wait_time}s, proceeding anyway")
				else:
					self.logger.info(f"Previous training completed with status: {current_status}")

			# Start new training
			self.logger.info(f"Starting training for group: {target_group_id}")
			self.face_admin_client.large_person_group.begin_train(target_group_id)

			# Update training state
			self.training_states[target_group_id] = True
			return True

		except Exception as e:
			self.logger.error(f"Error training group {target_group_id}: {e}")
			return False

	def get_training_status_simple(self, group_id: str) -> str:
		"""Get simple training status string for a group"""
		detailed_status = self.get_training_status_detailed(group_id)
		return detailed_status.get('status', 'unknown')

	def save_face_image_comprehensive(self, face_image: np.ndarray, person_id: str, face_id: str,
									 group_type: str = "unknown", save_enabled: bool = True,
									 person_name: str = None, user_data: str = "",
									 enforce_max_faces: bool = True) -> str | None:
		"""
		Comprehensive face image saving method that handles both known and unknown faces

		Args:
			face_image: The face image to save
			person_id: ID of the person
			face_id: ID of the face
			group_type: "known" or "unknown"
			save_enabled: Whether saving is enabled
			person_name: Name of person (for known faces)
			user_data: Additional user data
			enforce_max_faces: Whether to enforce max faces per person limit
		"""
		if not save_enabled:
			return None

		try:
			# Create directory structure: saved_images/group_type/person_id/
			person_dir = self.face_images_dir / group_type / person_id
			person_dir.mkdir(parents=True, exist_ok=True)

			# For known faces, save person info to JSON file
			if group_type == "known" and person_name:
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

			# Check if we already have max images for this person (mainly for unknown faces)
			if enforce_max_faces:
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

			self.logger.info(f"Saved {group_type} face image: {file_path}")
			return str(file_path)

		except Exception as e:
			self.logger.error(f"Error saving {group_type} face image: {e}")
			return None



	def get_person_images(self, person_id: str, group_type: str = "unknown") -> List[str]:
		"""Get list of saved image paths for a person in specified group type"""
		try:
			person_dir = self.face_images_dir / group_type / person_id
			if not person_dir.exists():
				return []

			# Get all jpg files in person directory
			image_files = list(person_dir.glob("*.jpg"))
			return [str(f) for f in image_files]

		except Exception as e:
			self.logger.error(f"Error getting {group_type} person images: {e}")
			return []

	def get_person_info_from_files(self, person_id: str, group_type: str = "known") -> Dict[str, Any] | None:
		"""Get person info from JSON file for specified group type"""
		try:
			person_dir = self.face_images_dir / group_type / person_id
			json_file = person_dir / "person_info.json"

			if not json_file.exists():
				return None

			with open(json_file, 'r') as f:
				return json.load(f)

		except Exception as e:
			self.logger.error(f"Error getting {group_type} person info: {e}")
			return None

	def get_person_info_from_api(self, person_id: str, group_id: str) -> Dict[str, Any] | None:
		"""Get person information from Azure API for specified group"""
		try:
			person = self.face_admin_client.large_person_group.get_person(group_id, person_id)
			if person:
				return {
					'person_id': person.person_id,
					'name': person.name,
					'user_data': person.user_data,
					'persisted_face_ids': person.persisted_face_ids
				}
			return None

		except Exception as e:
			self.logger.error(f"Error getting person info for {person_id} from API: {e}")
			return None



	def identify_faces(self, face_ids: List[str], group_id: str, confidence_threshold: float = 0.6) -> List[Any]:
		"""Common face identification method for both known and unknown groups"""
		try:
			self.logger.debug(f"Identifying {len(face_ids)} faces against group: {group_id}")

			identification_results = self.face_client.identify_from_large_person_group(
				face_ids=face_ids,
				large_person_group_id=group_id,
				confidence_threshold=confidence_threshold
			)

			self.logger.debug(f"Identification completed: {len(identification_results)} results")
			return identification_results

		except Exception as e:
			self.logger.error(f"Error identifying faces in group {group_id}: {e}")
			return []

	def get_training_status_detailed(self, group_id: str) -> Dict[str, Any]:
		"""Get detailed training status for a specific group - consolidated method"""
		try:
			status: FaceTrainingResult = self.face_admin_client.large_person_group.get_training_status(group_id)
			return {
				"group_id": group_id,
				"status": status.status.value if status.status else "unknown",
				"created_time": status.created_date_time.isoformat() if status.created_date_time else None,
				"last_action_time": status.last_action_date_time.isoformat() if status.last_action_date_time else None,
				"last_successful_training_time": status.last_successful_training_date_time.isoformat() if status.last_successful_training_date_time else None,
				"message": status.message or "",
				"training": self.training_states.get(group_id, False)  # Include local training state
			}
		except Exception as e:
			self.logger.error(f"Error getting training status for group {group_id}: {e}")
			return {
				"group_id": group_id,
				"status": "error",
				"message": str(e),
				"training": self.training_states.get(group_id, False),
				"created_time": None,
				"last_action_time": None,
				"last_successful_training_time": None
			}



	def get_group_statistics_detailed(self, group_id: str, group_name: str = None) -> Dict[str, Any]:
		"""Get detailed statistics for a specific group - consolidated method"""
		try:
			persons = self.face_admin_client.large_person_group.list_persons(
				large_person_group_id=group_id
			)

			total_persons = len(persons)
			total_faces = 0

			for person in persons:
				try:
					faces = self.face_admin_client.large_person_group.list_person_faces(
						large_person_group_id=group_id,
						person_id=person.person_id
					)
					total_faces += len(faces)
				except Exception as e:
					self.logger.debug(f"Error counting faces for person {person.person_id}: {e}")

			return {
				"group_id": group_id,
				"group_name": group_name or group_id,
				"total_persons": total_persons,
				"total_faces": total_faces,
				"average_faces_per_person": total_faces / total_persons if total_persons > 0 else 0,
				"training_status": self.get_training_status_simple(group_id)
			}
		except Exception as e:
			self.logger.error(f"Error getting statistics for group {group_id}: {e}")
			return {
				"group_id": group_id,
				"group_name": group_name or group_id,
				"total_persons": 0,
				"total_faces": 0,
				"average_faces_per_person": 0,
				"training_status": "unknown"
			}



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

			self.train_group()
			return True
			
		except Exception as e:
			self.logger.error(f"Error cleaning up group {group_id}: {e}")
			return False


