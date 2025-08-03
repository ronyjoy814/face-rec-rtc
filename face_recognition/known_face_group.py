# coding: utf-8

"""
Known Face Group Management

Manages known persons and their face data using Azure AI Vision Face API.
Provides comprehensive face recognition capabilities including person management,
face enrollment, training coordination, and identification services.

Key Features:
    - Person registration and management
    - Face enrollment and training
    - Real-time face identification
    - Training status monitoring
    - Group statistics and reporting
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from azure.ai.vision.face.models import FaceDetectionModel
from .base_group_manager import BaseGroupManager


class KnownFaceGroup(BaseGroupManager):
    """
    Known Face Group Manager

    Manages a collection of known persons and their associated face data.
    Provides functionality for person registration, face enrollment, training,
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
        save_known_images: bool = True,
        cleanup_known_group: bool = False,
        logger: Optional[Any] = None,
        log_level: int = logging.INFO
    ):
        """Initialize known face group manager"""
        super().__init__(endpoint, api_key, max_faces_per_person, face_similarity_threshold, logger, log_level)

        self.base_group_id = group_id
        self.base_group_name = group_name
        self.save_known_images = save_known_images
        self.cleanup_known_group = cleanup_known_group

        # Single known group ID
        self.group_id = f"{group_id}_known"

        # Initialize group
        self._initialize_group()
    
    def _initialize_group(self):
        """Initialize the known group"""
        try:
            group_name = f"{self.base_group_name} - Known"
            self.create_group_if_not_exists(self.group_id, group_name)
            self.logger.info(f"Initialized known group: {self.group_id}")
            if self.cleanup_known_group:
                self.cleanup_group()
        except Exception as e:
            self.logger.error(f"Error initializing known group: {e}")
            raise
    
    def get_group_id(self) -> str:
        """Get the known group ID"""
        return self.group_id
    
    def identify_faces(self, face_ids: List[str], confidence_threshold: float = 0.6) -> List[Any]:
        """Identify faces against known group using common base method"""
        return super().identify_faces(face_ids, self.group_id, confidence_threshold)
    
    def add_person(self, person_name: str, user_data: str = None) -> Optional[str]:
        """Add a new person to the known group"""
        try:
            self.logger.info(f"Adding person '{person_name}' to known group")
            
            person = self.face_admin_client.large_person_group.create_person(
                large_person_group_id=self.group_id,
                name=person_name,
                user_data=user_data or f"Known person: {person_name}"
            )
            
            if person and person.person_id:
                self.logger.info(f"Successfully added person '{person_name}' with ID: {person.person_id}")
                return person.person_id
            else:
                self.logger.error(f"Failed to add person '{person_name}' - no person ID returned")
                return None
                
        except Exception as e:
            self.logger.error(f"Error adding person '{person_name}': {e}")
            return None
    
    def add_face_to_person(self, person_id: str, face_image: np.ndarray, user_data: str = None) -> bool:
        """Add a face image to a known person with timestamp"""
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
                user_data=user_data or f"Face for person {person_id}",
                detection_model=FaceDetectionModel.DETECTION03
            )
            
            if face_result and face_result.persisted_face_id:
                self.logger.info(f"Successfully added face to person {person_id}")
                
                # Save image if enabled using structured file system
                if self.save_known_images and hasattr(self, 'group_manager'):
                    # Get person name for saving
                    person_name = self._get_person_name(person_id)
                    if person_name:
                        self.group_manager.save_person_face_crop(
                            person_id, person_name, face_image, True, user_data
                        )
                
                return True
            else:
                self.logger.error(f"Failed to add face to person {person_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding face to person {person_id}: {e}")
            return False

    def add_face_with_similarity_check(self, face_image: np.ndarray, person_id: str = None,
                                     confidence_threshold: float = 0.8,
                                     time_gap_seconds: int = 0, max_faces_per_person: int = 5) -> Optional[str]:
        """
        Add face to known group with similarity checking and time gap validation

        Args:
            face_image: Face image to add
            person_id: Specific person ID to add face to (if None, will identify first)
            confidence_threshold: Minimum confidence for face matching
            time_gap_seconds: Minimum time gap between faces (default: 0 seconds)
            max_faces_per_person: Maximum faces per person (default: 5)

        Returns:
            Person ID if face was added, None otherwise
        """
        try:
            from datetime import datetime

            current_time = datetime.now()

            # If person_id is provided, use it directly; otherwise identify
            target_person_id = person_id

            if not target_person_id:
                # Try to identify against existing known persons
                identified_faces = self.identify_faces([face_image])

                for face_result in identified_faces:
                    if (face_result.get('confidence', 0) >= confidence_threshold and
                        face_result.get('person_id')):
                        target_person_id = face_result['person_id']
                        break

            if target_person_id:
                # Check if person has reached max faces limit
                person_faces = self.get_person_faces(target_person_id)
                if len(person_faces) >= max_faces_per_person:
                    self.logger.info(f"Known person {target_person_id} has reached max faces limit ({max_faces_per_person})")
                    return None

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
                                self.logger.info(f"Time gap too small ({time_diff}s < {time_gap_seconds}s) for known person {target_person_id}")
                                valid_time_gap = False
                                break
                        except Exception as e:
                            self.logger.warning(f"Could not parse face timestamp: {e}")

                if valid_time_gap:
                    # Add face to existing known person
                    face_user_data = f"Captured: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    if self.add_face_to_person(target_person_id, face_image, face_user_data):
                        person_info = self.get_person_info(target_person_id)
                        person_name = person_info.get('name', 'Unknown') if person_info else 'Unknown'
                        self.logger.info(f"Added face to existing known person '{person_name}' ({target_person_id})")
                        return target_person_id

            # No similar known person found
            self.logger.info("No similar known person found for face")
            return None

        except Exception as e:
            self.logger.error(f"Error in known group similarity-based face addition: {e}")
            return None

    def get_person_faces(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all faces for a specific known person using base class method"""
        return super().get_person_faces_from_api(person_id, self.group_id)
    
    def get_person_info(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a person using base class method"""
        return super().get_person_info_from_api(person_id, self.group_id)
    
    def list_persons(self) -> List[Dict[str, Any]]:
        """List all persons in the known group using base class method"""
        return super().list_persons_in_group(self.group_id)
    
    def delete_person(self, person_id: str) -> bool:
        """Delete a person from the known group and clean up their folder"""
        try:
            # Delete from Azure API using base class method
            success = super().delete_person_with_training(self.group_id, person_id, "known group")
            
            if success and hasattr(self, 'group_manager') and self.group_manager:
                # Clean up the person's folder from file system
                self.group_manager._cleanup_person_files(person_id, True)
                self.logger.info(f"Cleaned up folder for deleted known person: {person_id}")
            
            return success
        except Exception as e:
            self.logger.error(f"Error deleting known person {person_id}: {e}")
            return False
    
    def train_group(self) -> bool:
        """Train the known group using base class method"""
        return super().train_group(self.group_id)
    
    def get_training_status(self) -> str:
        """Get training status of the known group using base class method"""
        return super().get_training_status_simple(self.group_id)
    


    def _get_person_name(self, person_id: str) -> Optional[str]:
        """Get person name by person ID using base class method"""
        return super().get_person_name_from_api(person_id, self.group_id)

    def set_group_manager(self, group_manager):
        """Set reference to the parent group manager for file operations using base class method"""
        super().set_group_manager_reference(group_manager)

    def get_group_stats(self) -> Dict[str, Any]:
        """Get statistics about the known group using base class method"""
        return super().get_group_statistics_detailed(self.group_id, self.base_group_name)

    def transfer_from_unknown_group(self, unknown_group, person_id: str, person_details: Dict[str, Any]) -> bool:
        """Transfer an unknown person to known group with their details"""
        try:
            name = person_details.get("name", "Unknown")
            user_data = person_details.get("user_data", "")

            # Add person to known group
            new_person_id = self.add_person(name, user_data)
            if not new_person_id:
                self.logger.error(f"Failed to add person {name} to known group")
                return False

            # Transfer images from unknown to known using base class method
            success = self.transfer_unknown_to_known(person_id, new_person_id, name, user_data)
            if success:
                self.logger.info(f"Successfully transferred unknown person {person_id} to known person {new_person_id} ({name})")
                return True
            else:
                self.logger.error(f"Failed to transfer images for person {person_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error transferring unknown person {person_id}: {e}")
            return False

    def process_pending_unknown_transfers(self, unknown_group) -> int:
        """Process all pending unknown persons that have details and transfer them to known group"""
        try:
            pending_persons = unknown_group.get_pending_known_persons()
            transferred_count = 0

            for person_id, person_details in pending_persons.items():
                if self.transfer_from_unknown_group(unknown_group, person_id, person_details):
                    unknown_group.clear_pending_known_person(person_id)
                    transferred_count += 1

            if transferred_count > 0:
                self.logger.info(f"Transferred {transferred_count} unknown persons to known group")

            return transferred_count

        except Exception as e:
            self.logger.error(f"Error processing pending unknown transfers: {e}")
            return 0
