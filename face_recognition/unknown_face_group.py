# coding: utf-8

"""
Unknown Face Group Management

Manages unidentified persons and their face data using Azure AI Vision Face API.
Provides functionality for automatic person creation, face enrollment, training,
and identification services for previously unknown individuals.

Key Features:
    - Automatic person registration for unknown faces
    - Face enrollment and training
    - Person identification and matching
    - Training status monitoring
    - Group statistics and reporting
"""

import os
import cv2
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from .base_group_manager import BaseGroupManager


class UnknownFaceGroup(BaseGroupManager):
    """
    Unknown Face Group Manager

    Manages a collection of unidentified persons and their associated face data.
    Provides functionality for automatic person creation, face enrollment, training,
    and identification using Azure AI Vision Face API.
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        group_id: str,
        group_name: str,
        max_faces_per_person: int = 5,
        face_similarity_threshold: float = 0.8,
        auto_train_unknown: bool = True,
        unknown_confidence_threshold: float = 0.8,
        save_unknown_images: bool = True,
        cleanup_unknown_images: bool = False,
        logger: Optional[Any] = None,
        log_level: int = logging.INFO
    ):
        """Initialize unknown face group manager"""
        super().__init__(endpoint, api_key, max_faces_per_person, face_similarity_threshold, logger, log_level)

        self.base_group_id = group_id
        self.base_group_name = group_name
        self.auto_train_unknown = auto_train_unknown
        self.unknown_confidence_threshold = unknown_confidence_threshold
        self.save_unknown_images = save_unknown_images

        # Single unknown group ID
        self.group_id = f"{group_id}_unknown"

        # Unknown person counter
        self.unknown_person_counter = 1

        # Storage for unknown persons with name and user_data details
        self.pending_known_persons = {}  # {person_id: {"name": str, "user_data": str, "timestamp": datetime}}

        # Training coordination flags and data
        self.known_group_training_flag = False
        self.pending_transfers = {}  # person_name -> {known_name, user_data}
        self.transferred_persons = set()  # Track persons marked for cleanup
        self.persons_for_cleanup = set()  # Persons ready for cleanup after training

        self.cleanup_unknown_images_flag = cleanup_unknown_images

        # Initialize group
        self._initialize_group()
    
    def _initialize_group(self):
        """Initialize the unknown group"""
        try:
            group_name = f"{self.base_group_name} - Unknown"
            self.create_group_if_not_exists(self.group_id, group_name)

            # Clear all existing persons in the unknown group to avoid accumulation
            self._clear_all_persons()
            if self.cleanup_unknown_images_flag:
                self._cleanup_unknown_images()

            self.logger.info(f"Initialized unknown group: {self.group_id}")
        except Exception as e:
            self.logger.error(f"Error initializing unknown group: {e}")
            raise

    def _clear_all_persons(self):
        """Clear all existing persons from the unknown group to avoid accumulation"""
        try:
            # Get all existing persons
            existing_persons = self.list_persons()

            if existing_persons:
                self.logger.info(f"Clearing {len(existing_persons)} existing persons from unknown group")

                # Delete each person
                for person in existing_persons:
                    try:
                        person_id = person['person_id']
                        self.face_admin_client.large_person_group.delete_person(
                            large_person_group_id=self.group_id,
                            person_id=person_id
                        )
                        self.logger.debug(f"Deleted person {person_id} from unknown group")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete person {person_id}: {e}")

                self.logger.info("Successfully cleared all persons from unknown group")
                self.train_group()
                self.logger.info("Training unknown group after clearing persons")
            else:
                self.logger.info("No existing persons found in unknown group")
        except Exception as e:
            self.logger.error(f"Error clearing persons from unknown group: {e}")

    def clear_all_persons(self):
        """Public method to clear all existing persons from the unknown group"""
        self.logger.info("Manual clear all persons requested for unknown group")
        self._clear_all_persons()

    def get_group_id(self) -> str:
        """Get the unknown group ID"""
        return self.group_id
    
    def identify_faces(self, face_ids: List[str], confidence_threshold: float = 0.6) -> List[Any]:
        """Identify faces against unknown group using common base method"""
        return super().identify_faces(face_ids, self.group_id, confidence_threshold)
    
    def add_unknown_person(self, face_image: np.ndarray, user_data: str = None) -> Optional[str]:
        """Add a new unknown person to the group with auto-generated person ID"""
        try:
            from datetime import datetime

            # Generate person name as person_1, person_2, etc.
            # Note: get_next_person_number() now handles the increment
            person_number = self.get_next_person_number()
            person_name = f"person_{person_number}"

            # Create user data with creation timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            person_user_data = user_data or f"Created: {current_time}"

            self.logger.info(f"Adding unknown person '{person_name}' to unknown group")

            # Create person
            person = self.face_admin_client.large_person_group.create_person(
                large_person_group_id=self.group_id,
                name=person_name,
                user_data=person_user_data
            )

            if person and person.person_id:
                # Add face to person with timestamp
                face_user_data = f"Captured: {current_time}"
                if self.add_face_to_person(person.person_id, face_image, face_user_data):
                    self.logger.info(f"Successfully added unknown person '{person_name}' with ID: {person.person_id}")

                    # Trigger training if enabled
                    if self.auto_train_unknown:
                        self.train_group()

                    return person.person_id
                else:
                    # Clean up person if face addition failed
                    self.delete_person(person.person_id)
                    return None
            else:
                self.logger.error(f"Failed to add unknown person '{person_name}' - no person ID returned")
                return None

        except Exception as e:
            self.logger.error(f"Error adding unknown person: {e}")
            return None
    
    def add_face_to_person(self, person_id: str, face_image: np.ndarray, user_data: str = None) -> bool:
        """Add a face image to an unknown person with timestamp"""
        try:
            from datetime import datetime

            # Add timestamp to user data if not provided
            if user_data is None:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                user_data = f"Captured: {current_time}"

            # Convert image to bytes
            face_bytes = self._image_to_bytes(face_image)
            if not face_bytes:
                self.logger.error("Failed to convert face image to bytes")
                return False
            
            # Add face to person
            face_result = self.face_admin_client.large_person_group.add_face(
                large_person_group_id=self.group_id,
                person_id=person_id,
                image_content=face_bytes,
                user_data=user_data or f"Face for unknown person {person_id}",
                detection_model=FaceDetectionModel.DETECTION03
            )
            
            if face_result and face_result.persisted_face_id:
                self.logger.info(f"Successfully added face to unknown person {person_id}")
                
                # Save image if enabled using structured file system
                if self.save_unknown_images and hasattr(self, 'group_manager'):
                    # Get person name for saving
                    person_name = self._get_person_name(person_id)
                    if person_name:
                        self.group_manager.save_person_face_crop(
                            person_id, person_name, face_image, False, user_data
                        )
                
                return True
            else:
                self.logger.error(f"Failed to add face to unknown person {person_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding face to unknown person {person_id}: {e}")
            return False

    def add_face_with_similarity_check(self, face_image: np.ndarray, person_id: str = None,
                                     similarity_threshold: float = 0.8, confidence_threshold: float = 0.8,
                                     time_gap_seconds: int = 0, max_faces_per_person: int = 5) -> Optional[str]:
        """
        Add face to unknown group with similarity checking and time gap validation

        Args:
            face_image: Face image to add
            similarity_threshold: Minimum similarity to consider a match
            confidence_threshold: Minimum confidence for face matching
            time_gap_seconds: Minimum time gap between faces (default: 5 seconds)
            max_faces_per_person: Maximum faces per person (default: 5)

        Returns:
            Person ID if face was added, None otherwise
        """
        try:
            from datetime import datetime

            current_time = datetime.now()

            # First detect face to get face_id for identification
            face_bytes = self._image_to_bytes(face_image)
            if not face_bytes:
                self.logger.error("Failed to convert face image to bytes for similarity check")
                return self.add_unknown_person(face_image)

            # Detect face to get face_id
            detected_faces = self.face_client.detect(
                face_bytes,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True
            )

            if not detected_faces:
                self.logger.debug("No faces detected in image for similarity check, creating new unknown person")
                return self.add_unknown_person(face_image)

            face_id = str(detected_faces[0].face_id)

            # Try to identify against existing unknown persons with a more restrictive threshold
            # Use a higher threshold for unknown person matching to avoid merging different people
            unknown_similarity_threshold = max(confidence_threshold, 0.995)  # At least 0.995 for unknown matching
            identified_faces = self.identify_faces([face_id], unknown_similarity_threshold)

            for face_result in identified_faces:
                if (hasattr(face_result, 'candidates') and face_result.candidates and
                    len(face_result.candidates) > 0):
                    
                    candidate = face_result.candidates[0]
                    if candidate.confidence >= unknown_similarity_threshold:
                        matched_person_id = candidate.person_id

                        # Check if person has reached max faces limit
                        person_faces = self.get_person_faces(matched_person_id)
                        if len(person_faces) >= max_faces_per_person:
                            self.logger.info(f"Person {matched_person_id} has reached max faces limit ({max_faces_per_person})")
                            continue

                        # Check time gap with existing faces
                        valid_time_gap = True
                        for face_info in person_faces:
                            face_user_data = face_info.get('user_data', '')
                            if 'Captured:' in face_user_data:
                                try:
                                    face_time_str = face_user_data.split('Captured: ')[1]
                                    face_time = datetime.strptime(face_time_str, "%Y-%m-%d %H:%M:%S")
                                    time_diff = abs((current_time - face_time).total_seconds())

                                    if time_diff < time_gap_seconds:
                                        self.logger.info(f"Time gap too small ({time_diff}s < {time_gap_seconds}s) for person {matched_person_id}")
                                        valid_time_gap = False
                                        break
                                except Exception as e:
                                    self.logger.warning(f"Could not parse face timestamp: {e}")

                        if valid_time_gap:
                            # Add face to existing person
                            face_user_data = f"Captured: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                            if self.add_face_to_person(matched_person_id, face_image, face_user_data):
                                self.logger.info(f"Added face to existing unknown person {matched_person_id} (similarity: {candidate.confidence:.3f})")
                                return matched_person_id

            # No similar person found, create new unknown person
            return self.add_unknown_person(face_image)

        except Exception as e:
            self.logger.error(f"Error in similarity-based face addition: {e}")
            return self.add_unknown_person(face_image)

    def get_person_faces(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all faces for a specific person using base class method"""
        return super().get_person_faces_from_api(person_id, self.group_id)
    
    def update_unknown_person(self, person_id: str, new_name: str, designation: str = None) -> bool:
        """Update an unknown person with real name and designation"""
        try:
            user_data = f"Real name: {new_name}"
            if designation:
                user_data += f", Designation: {designation}"
            
            self.face_admin_client.large_person_group.update_person(
                large_person_group_id=self.group_id,
                person_id=person_id,
                name=new_name,
                user_data=user_data
            )
            
            self.logger.info(f"Updated unknown person {person_id} to '{new_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating unknown person {person_id}: {e}")
            return False
    
    def get_person_info(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an unknown person using base class method"""
        return super().get_person_info_from_api(person_id, self.group_id)
    
    def list_persons(self) -> List[Dict[str, Any]]:
        """List all persons in the unknown group using base class method"""
        return super().list_persons_in_group(self.group_id)
    
    def delete_person(self, person_id: str) -> bool:
        """Delete an unknown person from the group using base class method"""
        return super().delete_person_with_training(self.group_id, person_id, "unknown group")

    def get_person_id_by_name(self, person_name: str) -> str | None:
        """Get person ID by person name"""
        try:
            persons = self.face_admin_client.large_person_group.get_persons(self.group_id)

            for person in persons:
                if person.name == person_name:
                    return person.person_id

            self.logger.debug(f"Person with name '{person_name}' not found")
            return None

        except Exception as e:
            self.logger.error(f"Error getting person ID by name: {e}")
            return None

    def get_person_face_images(self, person_id: str) -> List[np.ndarray]:
        """Get all face images for a person"""
        try:
            # Note: We get face images from the file system since Azure doesn't store the actual image data

            face_images = []

            # Try to get images from structured file system first
            if hasattr(self, 'group_manager') and self.group_manager:
                person_folder = os.path.join(str(self.group_manager.base_save_directory), "unknown", person_id)
            else:
                person_folder = os.path.join("face_recognition/saved_images", "unknown", person_id)
            if os.path.exists(person_folder):
                for filename in os.listdir(person_folder):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_folder, filename)
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                face_images.append(image)
                        except Exception as e:
                            self.logger.debug(f"Error loading image {image_path}: {e}")

            self.logger.info(f"Retrieved {len(face_images)} face images for person {person_id}")
            return face_images

        except Exception as e:
            self.logger.error(f"Error getting person face images: {e}")
            return []
    
    def train_group(self) -> bool:
        """Train the unknown group using base class method"""
        return super().train_group(self.group_id)
    
    def get_training_status(self) -> str:
        """Get training status of the unknown group using base class method"""
        return super().get_training_status_simple(self.group_id)
    


    def _cleanup_unknown_images(self):
        """Clean up all saved unknown images"""
        try:
            import shutil

            if self.unknown_images_dir.exists():
                # Remove all unknown image directories and files
                for item in self.unknown_images_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                        self.logger.debug(f"Removed unknown person directory: {item}")
                    elif item.is_file():
                        item.unlink()
                        self.logger.debug(f"Removed unknown image file: {item}")

                self.logger.info("Successfully cleaned up all unknown images")
            else:
                self.logger.debug("Unknown images directory does not exist, nothing to clean")

        except Exception as e:
            self.logger.error(f"Error cleaning up unknown images: {e}")

    def cleanup_unknown_images(self):
        """Public method to manually clean up unknown images"""
        self.logger.info("Manual cleanup of unknown images requested")
        self._cleanup_unknown_images()

    def set_person_details(self, person_id: str, name: str, user_data: str = "") -> bool:
        """Set name and user_data for an unknown person, marking them for transfer to known group"""
        try:
            from datetime import datetime

            if person_id not in self.pending_known_persons:
                self.pending_known_persons[person_id] = {
                    "name": name,
                    "user_data": user_data,
                    "timestamp": datetime.now()
                }
                self.logger.info(f"Set details for unknown person {person_id}: {name}")
                return True
            else:
                # Update existing details
                self.pending_known_persons[person_id].update({
                    "name": name,
                    "user_data": user_data,
                    "timestamp": datetime.now()
                })
                self.logger.info(f"Updated details for unknown person {person_id}: {name}")
                return True

        except Exception as e:
            self.logger.error(f"Error setting person details for {person_id}: {e}")
            return False

    def get_pending_known_persons(self) -> Dict[str, Dict[str, Any]]:
        """Get all unknown persons that have name and user_data details set"""
        return self.pending_known_persons.copy()

    def clear_pending_known_person(self, person_id: str):
        """Remove a person from pending known persons list after successful transfer"""
        if person_id in self.pending_known_persons:
            del self.pending_known_persons[person_id]
            self.logger.info(f"Cleared pending known person: {person_id}")

    def has_pending_known_persons(self) -> bool:
        """Check if there are any unknown persons with details waiting to be transferred"""
        return len(self.pending_known_persons) > 0

    def get_group_stats(self) -> Dict[str, Any]:
        """Get statistics about the unknown group using base class method"""
        return super().get_group_statistics_detailed(self.group_id, self.base_group_name)

    # Training Coordination Methods for Complete Workflow

    def set_known_group_training_flag(self, is_training: bool):
        """Set flag indicating known group training status"""
        self.known_group_training_flag = is_training
        if is_training:
            self.logger.info("Known group training flag set - unknown group alerted")
        else:
            self.logger.info("Known group training flag cleared")

    def get_pending_training_data(self) -> Dict[str, Any]:
        """Get any pending training data for known group"""
        return {
            'pending_transfers': self.pending_transfers,
            'has_pending': len(self.pending_transfers) > 0
        }

    def mark_person_for_transfer(self, person_name: str, known_name: str, user_data: str):
        """Mark a person for transfer to known group"""
        self.pending_transfers[person_name] = {
            'known_name': known_name,
            'user_data': user_data,
            'timestamp': datetime.now()
        }
        self.logger.info(f"Marked {person_name} for transfer to known group as {known_name}")

    def has_pending_transfers(self) -> bool:
        """Check if there are pending transfers"""
        return len(self.pending_transfers) > 0

    def get_person_data_for_transfer(self, person_name: str) -> Optional[Dict[str, Any]]:
        """Get person data for transfer to known group"""
        try:
            # Find person by name
            persons = self.list_persons()
            target_person = None

            for person in persons:
                person_info = self.get_person_info(person.person_id)
                if person_info and person_info.get('name') == person_name:
                    target_person = person
                    break

            if not target_person:
                self.logger.error(f"Person {person_name} not found for transfer")
                return None

            # Get all faces for this person
            faces_data = []
            try:
                # Get person info which contains persisted_face_ids
                person_info = self.face_admin_client.large_person_group.get_person(
                    large_person_group_id=self.group_id,
                    person_id=target_person.person_id
                )

                if person_info and person_info.persisted_face_ids:
                    for face_id in person_info.persisted_face_ids:
                        try:
                            # Get individual face details
                            face = self.face_admin_client.large_person_group.get_face(
                                large_person_group_id=self.group_id,
                                person_id=target_person.person_id,
                                persisted_face_id=face_id
                            )
                            faces_data.append({
                                'face_id': face.persisted_face_id,
                                'user_data': face.user_data,
                                'image': None  # Would contain actual image data
                            })
                        except Exception as face_error:
                            self.logger.debug(f"Could not get face {face_id}: {face_error}")

            except Exception as e:
                self.logger.warning(f"Could not get faces for person {person_name}: {e}")

            return {
                'person_id': target_person.person_id,
                'person_name': person_name,
                'user_data': target_person.user_data,
                'faces': faces_data
            }

        except Exception as e:
            self.logger.error(f"Error getting person data for transfer: {e}")
            return None

    def mark_person_as_transferred(self, person_name: str):
        """Mark person as transferred to known group"""
        self.transferred_persons.add(person_name)
        # Remove from pending transfers
        if person_name in self.pending_transfers:
            del self.pending_transfers[person_name]
        self.logger.info(f"Marked {person_name} as transferred")

    def schedule_cleanup_after_training(self):
        """Schedule cleanup of transferred persons after training"""
        self.persons_for_cleanup.update(self.transferred_persons)
        self.logger.info(f"Scheduled {len(self.transferred_persons)} persons for cleanup after training")

    def cleanup_transferred_persons(self) -> int:
        """
        Clean up persons that have been transferred and trained

        Returns:
            int: Number of persons cleaned up
        """
        cleaned_count = 0

        try:
            persons_to_cleanup = list(self.persons_for_cleanup)

            for person_name in persons_to_cleanup:
                try:
                    # Find and delete person
                    persons = self.list_persons()
                    for person in persons:
                        person_info = self.get_person_info(person.person_id)
                        if person_info and person_info.get('name') == person_name:
                            # Delete person (this also deletes all their faces)
                            self.delete_person(person.person_id)
                            cleaned_count += 1
                            self.logger.info(f"Cleaned up transferred person: {person_name}")
                            break

                    # Remove from tracking sets
                    self.persons_for_cleanup.discard(person_name)
                    self.transferred_persons.discard(person_name)

                except Exception as e:
                    self.logger.error(f"Error cleaning up person {person_name}: {e}")

            if cleaned_count > 0:
                self.logger.info(f"Unknown group cleanup complete: removed {cleaned_count} transferred persons")

            return cleaned_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0

    def get_next_person_number(self) -> int:
        """Get the next available person number for unknown persons"""
        # Use the internal counter and increment it to ensure unique numbers
        current_number = self.unknown_person_counter
        self.unknown_person_counter += 1
        return current_number

    def _get_person_name(self, person_id: str) -> Optional[str]:
        """Get person name by person ID using base class method"""
        return super().get_person_name_from_api(person_id, self.group_id)

    def set_group_manager(self, group_manager):
        """Set reference to the parent group manager for file operations using base class method"""
        super().set_group_manager_reference(group_manager)
