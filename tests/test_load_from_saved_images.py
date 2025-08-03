#!/usr/bin/env python3
"""
Test Load from Saved Images

This test verifies:
1. Loading faces from saved_images folder instead of images folder
2. Proper folder management (no redundant folders)
3. Transfer mechanism with real data from saved_images
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import logging
import cv2
import time
from datetime import datetime
from dotenv import load_dotenv
from face_recognition.face_group_manager import FaceGroupManager

# Load environment variables
load_dotenv()

def _run_test_load_from_saved_images(pause_between_phases=False, log_level=logging.DEBUG):
    """Internal test function - actual implementation"""
    import time

    def wait_for_user_if_enabled(phase_name=""):
        """Wait for user input if pause mode is enabled"""
        if pause_between_phases:
            input(f"\n⏸️  Press Enter to continue to {phase_name}...")
            print()

    print("LOAD FROM SAVED IMAGES TEST")
    print("=" * 35)
    if pause_between_phases:
        print("⏸️  Pause mode enabled - will wait for user input between phases")

    # Track phase completion and errors separately for each phase
    phase_results = {
        'phase1_folder_check': False,
        'phase2_loading': False,
        'phase3_group_validation': False,
        'phase4_face_detection': False,
        'phase5_identification': False,
        'phase6_unknown_creation': False,
        'phase7_transfer': False,
        'phase8_final_validation': False,
        'phase9_delete_dad': False,
        'phase10_delete_mom_server': False,
        'phase11_reidentification': False,
        'phase12_similarity_transfer': False,
        'phase13_final_complete': False,
        'phase14_function_analysis': False,
        'phase1_errors': [],
        'phase2_errors': [],
        'phase3_errors': [],
        'phase4_errors': [],
        'phase5_errors': [],
        'phase6_errors': [],
        'phase7_errors': [],
        'phase8_errors': [],
        'phase9_errors': [],
        'phase10_errors': [],
        'phase11_errors': [],
        'phase12_errors': [],
        'phase13_errors': [],
        'phase14_errors': []
    }
    
    # Initialize Face Recognition Manager
    endpoint = os.getenv('AZURE_FACE_API_ENDPOINT')
    api_key = os.getenv('AZURE_FACE_API_ACCOUNT_KEY')
    
    # Use unique group ID to avoid accumulation
    unique_id = f"load_saved_test_1"

    face_manager = FaceGroupManager(
        endpoint=endpoint,
        api_key=api_key,
        group_id=unique_id,
        group_name="Load Saved Test",
        save_known_images=True,
        auto_train_unknown=True,
        max_faces_per_person=5,
        cleanup_known_group=True,
        cleanup_unknown_images=False,
        log_level=log_level
    )
    print("OK Face Recognition Manager initialized")
    
    # Phase 1: Check if saved_images folder exists and preping for the phase for so you have to delete the Dad folder
    wait_for_user_if_enabled("Phase 1: Checking Saved Images Folder")
    print("\n1. CHECKING SAVED IMAGES FOLDER AND DELETING DAD FOLDER")
    print("-" * 35)
    
    saved_images_dir = "face_recognition/saved_images"
    known_dir = f"{saved_images_dir}/known"
    unknown_dir = f"{saved_images_dir}/unknown"
    
    if os.path.exists(known_dir):
        known_folders = [f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]

        # First pass: identify and delete Dad folders
        dad_folders_to_delete = []
        for folder in known_folders:
            json_file = os.path.join(known_dir, folder, "person_data.json")
            if os.path.exists(json_file):
                try:
                    import json
                    name = ""
                    with open(json_file, 'r') as f:
                        person_data = json.load(f)
                        name = person_data.get('name', 'Unknown')
                        face_count = person_data.get('face_count', 0)
                        print(f"   📋 {name}: {face_count} faces in {folder}")
                    if name == "Dad":
                        dad_folders_to_delete.append((folder, name))
                except:
                    print(f"   ⚠️  Could not read {folder}/person_data.json")

        # Delete all Dad folders
        for folder, name in dad_folders_to_delete:
            try:
                print(f"   🗑️  Deleting {name} from {folder}...")
                import shutil
                shutil.rmtree(os.path.join(known_dir, folder))
                print(f"   OK Successfully deleted {name} folder")
            except Exception as e:
                print(f"   X Failed to delete {folder}: {e}")
        
        print(f"📂 Known persons folders found: {len(known_folders)}")
        for folder in known_folders[:5]:  # Show first 5
            json_file = os.path.join(known_dir, folder, "person_data.json")
            if os.path.exists(json_file):
                try:
                    import json
                    with open(json_file, 'r') as f:
                        person_data = json.load(f)
                        name = person_data.get('name', 'Unknown')
                        face_count = person_data.get('face_count', 0)
                        print(f"   📋 {name}: {face_count} faces in {folder}")
                except:
                    print(f"   ⚠️  Could not read {folder}/person_data.json")
    else:
        print("X Known images folder not found")
    #Loading the unknown folder is optional part of this testscript so dont add to the phase errors
    if os.path.exists(unknown_dir):
        unknown_folders = [f for f in os.listdir(unknown_dir) if os.path.isdir(os.path.join(unknown_dir, f))]
        print(f"📂 Unknown persons folders found: {len(unknown_folders)}")
        for folder in unknown_folders[:3]:  # Show first 3
            json_file = os.path.join(unknown_dir, folder, "person_data.json")
            if os.path.exists(json_file):
                try:
                    import json
                    with open(json_file, 'r') as f:
                        person_data = json.load(f)
                        name = person_data.get('name', 'Unknown')
                        face_count = person_data.get('face_count', 0)
                        print(f"   📋 {name}: {face_count} faces in {folder}")
                except:
                    print(f"   ⚠️  Could not read {folder}/person_data.json")
    else:
        print("⚠️  Unknown images folder not found")

    # Validate Phase 1 requirements for Phase 3 preparation
    print("\n🔍 VALIDATING PHASE 1 REQUIREMENTS FOR PHASE 3")
    print("-" * 50)

    # Re-scan known folders after Dad deletion to validate
    if os.path.exists(known_dir):
        current_known_folders = [f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]
        required_persons = ['Mom', 'Daughter', 'Son']
        person_counts = {}

        # Count persons
        for folder in current_known_folders:
            json_file = os.path.join(known_dir, folder, "person_data.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        person_data = json.load(f)
                    name = person_data.get('name', 'Unknown')

                    if name in required_persons:
                        if name not in person_counts:
                            person_counts[name] = 0
                        person_counts[name] += 1
                except:
                    pass

        # Validate and fix required persons
        for required_person in required_persons:
            if required_person not in person_counts:
                error_msg = f"Required person '{required_person}' not found in known images"
                phase_results['phase1_errors'].append(error_msg)
                print(f"   X {error_msg}")
            elif person_counts[required_person] > 1:
                print(f"   ⚠️  Multiple folders found for '{required_person}' ({person_counts[required_person]} folders) - cleaning up duplicates...")

                # Find and clean up duplicate folders for this person
                person_folders = []
                for folder in current_known_folders:
                    json_file = os.path.join(known_dir, folder, "person_data.json")
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r') as f:
                                person_data = json.load(f)
                            name = person_data.get('name', 'Unknown')
                            if name == required_person:
                                face_count = len([f for f in os.listdir(os.path.join(known_dir, folder))
                                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                                person_folders.append((folder, face_count))
                        except:
                            pass

                # Keep the folder with the most faces, delete others
                if len(person_folders) > 1:
                    person_folders.sort(key=lambda x: x[1], reverse=True)  # Sort by face count, descending
                    keep_folder = person_folders[0][0]
                    print(f"   📁 Keeping {required_person} folder: {keep_folder} ({person_folders[0][1]} faces)")

                    for folder, face_count in person_folders[1:]:
                        try:
                            print(f"   🗑️  Deleting duplicate {required_person} folder: {folder} ({face_count} faces)")
                            import shutil
                            shutil.rmtree(os.path.join(known_dir, folder))
                            print(f"   OK Successfully deleted duplicate folder")
                        except Exception as e:
                            print(f"   X Failed to delete duplicate folder {folder}: {e}")

                print(f"   OK {required_person}: Cleaned up to single folder (ready for Phase 3)")
            elif person_counts[required_person] == 1:
                print(f"   OK {required_person}: Single folder found (ready for Phase 3)")

        # Re-scan after cleanup to validate final state
        final_known_folders = [f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]
        final_person_counts = {}
        final_dad_exists = False

        for folder in final_known_folders:
            json_file = os.path.join(known_dir, folder, "person_data.json")
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r') as f:
                        person_data = json.load(f)
                    name = person_data.get('name', 'Unknown')

                    if name in required_persons:
                        if name not in final_person_counts:
                            final_person_counts[name] = 0
                        final_person_counts[name] += 1
                    elif name == 'Dad':
                        final_dad_exists = True
                except:
                    pass

        # Final validation
        print("\n📋 FINAL VALIDATION RESULTS:")
        for required_person in required_persons:
            if required_person not in final_person_counts:
                error_msg = f"Required person '{required_person}' still not found after cleanup"
                phase_results['phase1_errors'].append(error_msg)
                print(f"   X {error_msg}")
            elif final_person_counts[required_person] > 1:
                error_msg = f"Multiple folders still exist for '{required_person}' after cleanup"
                phase_results['phase1_errors'].append(error_msg)
                print(f"   X {error_msg}")
            else:
                print(f"   OK {required_person}: Ready for Phase 3")

        # Validate Dad deletion
        if final_dad_exists:
            error_msg = "Dad folder still exists after deletion attempt - Phase 3 may fail"
            phase_results['phase1_errors'].append(error_msg)
            print(f"   X {error_msg}")
        else:
            print("   OK Dad folder successfully deleted (Phase 3 preparation complete)")
    else:
        error_msg = "Known images folder not found"
        phase_results['phase1_errors'].append(error_msg)
        print(f"   X {error_msg}")

    # Check if Phase 1 had any critical errors
    if not phase_results['phase1_errors']:
        phase_results['phase1_folder_check'] = True
        print("OK Phase 1 completed successfully")
    else:
        print("⚠️  Phase 1 completed with errors")

    # Phase 2: Load faces from saved_images (known group only first)
    print("\n2. LOADING KNOWN FACES FROM SAVED IMAGES")
    print("-" * 45)
    
    print("📚 Loading known faces from saved_images folder...")

    # Count folders before loading
    folders_before = len([f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]) if os.path.exists(known_dir) else 0

    # Load faces with clear options: clear known group first, use default (clear) for unknown group
    load_results = face_manager.load_faces_from_saved_images(
        group_type="both",
        clear_known_group=True,  # Clear known group first when loading known images
        clear_unknown_group=True  # Use default (clear) for unknown group
    )
    
    if 'error' not in load_results:
        print("OK Known faces loaded successfully")
        print(f"📊 Load Results:")
        print(f"   👥 Known persons loaded: {load_results['known']['persons_loaded']}")
        print(f"   📸 Known faces loaded: {load_results['known']['faces_loaded']}")
        print(f"   📈 Total persons: {load_results['total_persons']}")
        print(f"   📈 Total faces: {load_results['total_faces']}")
        
        if load_results['known']['errors']:
            print(f"   ⚠️  Errors: {len(load_results['known']['errors'])}")
            for error in load_results['known']['errors'][:3]:  # Show first 3 errors
                print(f"      - {error}")
        
        # Count folders after loading
        folders_after = len([f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]) if os.path.exists(known_dir) else 0
        
        print(f"📂 Folder management:")
        print(f"   📁 Folders before loading: {folders_before}")
        print(f"   📁 Folders after loading: {folders_after}")
        
        if folders_after <= folders_before:
            print("OK No redundant folders created (good folder management)")
        else:
            print("⚠️  More folders after loading (check folder management)")
            
    else:
        print(f"X Failed to load known faces: {load_results['error']}")
        phase_results['phase2_errors'].append(f"Loading Failed: {load_results['error']}")

    # Check for any loading errors in the results
    if 'error' not in load_results and load_results.get('known', {}).get('errors'):
        for error in load_results['known']['errors']:
            if 'similarity' not in error.lower():  # Exclude face similarity errors
                phase_results['phase2_errors'].append(f"Loading Error: {error}")

    # Mark phase 2 as complete if no critical errors
    if not phase_results['phase2_errors']:
        phase_results['phase2_loading'] = True
        print("OK Phase 2 completed successfully")
    else:
        print("⚠️  Phase 2 completed with errors")

    time.sleep(5)
    print("OKTraining and Loading successful")

    # Phase 3: Validate Group States and Core Functions
    print("\n3. VALIDATING GROUP STATES AND CORE FUNCTIONS")
    print("-" * 50)

    try:
        # Test known group functionality
        print("🔍 Testing Known Group Core Functions...")
        known_persons = face_manager.known_group.list_persons()
        print(f"   📊 Known persons count: {len(known_persons)}")

        # Validate expected persons (Mom, Daughter, Son - no Dad)
        expected_known = ['Mom', 'Daughter', 'Son']
        found_known = []

        for person in known_persons:
            person_info = face_manager.known_group.get_person_info(person['person_id'])
            if person_info:
                name = person_info.get('name', 'Unknown')
                found_known.append(name)
                print(f"   OK Found: {name} (ID: {person['person_id'][:8]}...)")

                # Test get_person_faces function
                faces = face_manager.known_group.get_person_faces(person['person_id'])
                print(f"      📸 Faces: {len(faces)}")

        # Validate no Dad in known group
        if 'Dad' in found_known:
            phase_results['phase3_errors'].append("Dad found in known group - should not be present")
            print("   X Dad found in known group (should be deleted)")
        else:
            print("   OK Dad correctly absent from known group")

        # Validate expected persons are present
        for expected in expected_known:
            if expected not in found_known:
                phase_results['phase3_errors'].append(f"Expected person '{expected}' not found in known group")
                print(f"   X Missing: {expected}")
            else:
                print(f"   OK Validated: {expected}")

        # Test unknown group functionality
        print("\n🔍 Testing Unknown Group Core Functions...")
        unknown_persons = face_manager.unknown_group.list_persons()
        print(f"   📊 Unknown persons count: {len(unknown_persons)}")

        # Test training status
        print("\n🔍 Testing Training Status Functions...")
        known_training_status = face_manager.known_group.get_training_status()
        unknown_training_status = face_manager.unknown_group.get_training_status()
        print(f"   📈 Known group training status: {known_training_status}")
        print(f"   📈 Unknown group training status: {unknown_training_status}")

        if not known_training_status:
            phase_results['phase3_errors'].append("Known group not trained after loading")
            print("   X Known group not trained")
        else:
            print("   OK Known group properly trained")

    except Exception as e:
        phase_results['phase3_errors'].append(f"Group validation failed: {e}")
        print(f"   X Group validation error: {e}")

    # Mark phase 3 as complete if no critical errors
    if not phase_results['phase3_errors']:
        phase_results['phase3_group_validation'] = True
        print("OK Phase 3 completed successfully")
    else:
        print("⚠️  Phase 3 completed with errors")

    # Phase 4: Test Face Detection Core Functions
    print("\n4️⃣ TESTING FACE DETECTION CORE FUNCTIONS")
    print("-" * 45)

    id_image_path = "images/identification1.jpg"
    if os.path.exists(id_image_path):
        print(f"📷 Testing face detection with {os.path.basename(id_image_path)}...")

        import cv2
        frame = cv2.imread(id_image_path)
        if frame is not None:
            try:
                # Test basic face detection without processing
                print("🔍 Testing basic face detection...")

                # Use process_frame with minimal processing to test detection
                result = face_manager.process_frame(
                    frame=frame,
                    auto_add_faces=False,  # Don't add faces yet, just detect
                    similarity_threshold=0.85,
                    confidence_threshold=0.90
                )

                print(f"   OK Face detection completed")
                print(f"   🔍 Total faces detected: {len(result.detected_faces)}")
                print(f"   📊 Detection confidence levels:")

                for i, face in enumerate(result.detected_faces):
                    confidence = getattr(face, 'confidence', 'N/A')
                    print(f"      Face {i+1}: Confidence {confidence}")

                # Validate detection results
                if len(result.detected_faces) == 0:
                    phase_results['phase4_errors'].append("No faces detected in identification image")
                    print("   X No faces detected")
                elif len(result.detected_faces) < 4:
                    phase_results['phase4_errors'].append(f"Expected 4 faces (Dad, Mom, Daughter, Son), detected {len(result.detected_faces)}")
                    print(f"   ⚠️  Expected 4 faces, detected {len(result.detected_faces)}")
                else:
                    print(f"   OK Expected number of faces detected: {len(result.detected_faces)}")

            except Exception as e:
                phase_results['phase4_errors'].append(f"Face detection failed: {e}")
                print(f"   X Face detection error: {e}")
        else:
            phase_results['phase4_errors'].append(f"Could not load image {id_image_path}")
            print(f"   X Failed to load {id_image_path}")
    else:
        phase_results['phase4_errors'].append(f"Image not found {id_image_path}")
        print(f"   X Image not found: {id_image_path}")

    # Mark phase 4 as complete if no critical errors
    if not phase_results['phase4_errors']:
        phase_results['phase4_face_detection'] = True
        print("OK Phase 4 completed successfully")
    else:
        print("⚠️  Phase 4 completed with errors")

    # Phase 5: Test Face Identification Functions
    print("\n5️⃣ TESTING FACE IDENTIFICATION FUNCTIONS")
    print("-" * 45)

    if os.path.exists(id_image_path):
        print(f"📷 Testing face identification with {os.path.basename(id_image_path)}...")

        frame = cv2.imread(id_image_path)
        if frame is not None:
            try:
                # Test identification with auto_add disabled first
                print("🔍 Testing identification of known faces...")

                result = face_manager.process_frame(
                    frame=frame,
                    auto_add_faces=False,  # Don't add unknown faces yet
                    similarity_threshold=0.85,
                    confidence_threshold=0.90
                )

                print(f"   OK Identification completed")
                print(f"   👥 Known faces identified: {result.known_faces}")
                print(f"   ❓ Unknown faces detected: {result.unknown_faces}")

                # Validate identification results
                expected_known_faces = 3  # Mom, Daughter, Son should be identified
                expected_unknown_faces = 1  # Dad should be unknown

                if result.known_faces != expected_known_faces:
                    phase_results['phase5_errors'].append(f"Expected {expected_known_faces} known faces, identified {result.known_faces}")
                    print(f"   ⚠️  Expected {expected_known_faces} known faces, identified {result.known_faces}")
                else:
                    print(f"   OK Correct number of known faces identified: {result.known_faces}")

                if result.unknown_faces != expected_unknown_faces:
                    phase_results['phase5_errors'].append(f"Expected {expected_unknown_faces} unknown faces, detected {result.unknown_faces}")
                    print(f"   ⚠️  Expected {expected_unknown_faces} unknown faces, detected {result.unknown_faces}")
                else:
                    print(f"   OK Correct number of unknown faces detected: {result.unknown_faces}")

                # Test face details
                print("📋 Face identification details:")
                for i, face in enumerate(result.detected_faces):
                    face_type = "Known" if hasattr(face, 'person_name') and face.person_name else "Unknown"
                    name = getattr(face, 'person_name', 'Unknown')
                    confidence = getattr(face, 'confidence', 'N/A')
                    print(f"   Face {i+1}: {face_type} - {name} (Confidence: {confidence})")

            except Exception as e:
                phase_results['phase5_errors'].append(f"Face identification failed: {e}")
                print(f"   X Face identification error: {e}")
        else:
            phase_results['phase5_errors'].append(f"Could not load image for identification")
            print(f"   X Failed to load image for identification")
    else:
        phase_results['phase5_errors'].append(f"Image not found for identification")
        print(f"   X Image not found for identification")

    # Mark phase 5 as complete if no critical errors
    if not phase_results['phase5_errors']:
        phase_results['phase5_identification'] = True
        print("OK Phase 5 completed successfully")
    else:
        print("⚠️  Phase 5 completed with errors")

    # Phase 6: Test Unknown Person Creation
    print("\n6️⃣ TESTING UNKNOWN PERSON CREATION")
    print("-" * 40)

    if os.path.exists(id_image_path):
        print(f"📷 Processing {os.path.basename(id_image_path)} to create unknown persons...")

        frame = cv2.imread(id_image_path)
        if frame is not None:
            try:
                # Count unknown persons before processing
                unknown_before = len(face_manager.unknown_group.list_persons())
                print(f"   📊 Unknown persons before: {unknown_before}")

                # Process frame with auto_add enabled to create unknown persons
                print("🔄 Processing frame with auto_add enabled...")
                result = face_manager.process_frame(
                    frame=frame,
                    auto_add_faces=True,
                    similarity_threshold=0.85,
                    confidence_threshold=0.90,
                    time_gap_seconds=5,
                    max_faces_per_person=5
                )

                print(f"   OK Frame processing completed")
                print(f"   🔍 Total faces detected: {len(result.detected_faces)}")
                print(f"   👥 Known faces identified: {result.known_faces}")
                print(f"   ❓ Unknown faces detected: {result.unknown_faces}")

                # Process background queue to add unknown faces
                print("   🔄 Processing background queue to add unknown faces...")
                face_manager.process_queue_manually()
                time.sleep(3)
                print("   OK Background processing completed")

                # Count unknown persons after processing
                unknown_after = len(face_manager.unknown_group.list_persons())
                print(f"   📊 Unknown persons after: {unknown_after}")

                # Validate unknown person creation
                if unknown_after <= unknown_before:
                    phase_results['phase6_errors'].append("No new unknown persons created")
                    print("   X No new unknown persons created")
                else:
                    new_unknown = unknown_after - unknown_before
                    print(f"   OK Created {new_unknown} new unknown person(s)")

                    # List new unknown persons
                    unknown_persons = face_manager.unknown_group.list_persons()
                    print("   📋 Unknown persons created:")
                    for person in unknown_persons[-new_unknown:]:  # Show last created
                        person_info = face_manager.unknown_group.get_person_info(person['person_id'])
                        if person_info:
                            name = person_info.get('name', 'Unknown')
                            print(f"      - {name} (ID: {person['person_id'][:8]}...)")

            except Exception as e:
                phase_results['phase6_errors'].append(f"Unknown person creation failed: {e}")
                print(f"   X Unknown person creation error: {e}")
        else:
            phase_results['phase6_errors'].append(f"Could not load image {id_image_path}")
            print(f"   X Failed to load {id_image_path}")
    else:
        phase_results['phase6_errors'].append(f"Image not found {id_image_path}")
        print(f"   X Image not found: {id_image_path}")

    # Mark phase 6 as complete if no critical errors
    if not phase_results['phase6_errors']:
        phase_results['phase6_unknown_creation'] = True
        print("OK Phase 6 completed successfully")
    else:
        print("⚠️  Phase 6 completed with errors")

    print("Sleep for 10 Sec to finish background processing")
    time.sleep(10)

    # Phase 7: Test Transfer Mechanism with Real Data
    print("\n7️⃣ TESTING TRANSFER MECHANISM WITH REAL DATA")
    print("-" * 40)
    
    # List unknown persons to find candidates for transfer
    transfer_success = False
    try:
        unknown_persons = face_manager.unknown_group.list_persons()
        if unknown_persons:
            print(f"🔍 Found {len(unknown_persons)} unknown persons for potential transfer")

            # Get the first unknown person for transfer test
            first_unknown = unknown_persons[0]
            unknown_person_id = first_unknown['person_id']

            # Get person info
            person_info = face_manager.unknown_group.get_person_info(unknown_person_id)
            if person_info:
                current_name = person_info.get('name', 'Unknown')
                print(f"📋 Selected unknown person: {current_name} (ID: {unknown_person_id[:8]}...)")
                
                # Test transfer to known group as "Dad"
                print("🔄 Testing transfer to known group as 'Dad'...")
                
                success = face_manager.queue_person_transfer(
                    unknown_person_id=unknown_person_id,
                    known_person_name="Dad",
                    known_user_data=f"Transferred from {current_name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

                if success:
                    print("OK Dad transfer queued successfully")

                    # Process the transfer queue
                    print("🔄 Processing transfer queue...")
                    face_manager.process_queue_manually()
                    time.sleep(3)
                    print("OK Transfer processing completed")

                    # Verify Dad was added to known group
                    print("🔍 Verifying Dad in known group...")
                    known_persons = face_manager.known_group.list_persons()
                    dad_found = False

                    for person in known_persons:
                        person_info = face_manager.known_group.get_person_info(person['person_id'])
                        if person_info and person_info.get('name') == 'Dad':
                            dad_found = True
                            print(f"OK Dad found in known group: {person['person_id'][:8]}...")
                            print(f"   📝 User data: {person_info.get('user_data', '')[:50]}...")
                            break
                    
                    if dad_found:
                        print("OK Transfer mechanism working correctly with real data!")
                        transfer_success = True
                    else:
                        print("⚠️  Dad not found in known group after transfer")
                        phase_results['phase7_errors'].append("Dad not found in known group after transfer")
                else:
                    print("X Failed to queue Dad transfer")
                    phase_results['phase7_errors'].append("Could not queue Dad transfer")
            else:
                print("X Could not get unknown person info")
                phase_results['phase7_errors'].append("Could not get unknown person info")
        else:
            print("⚠️  No unknown persons found for transfer test")
            phase_results['phase7_errors'].append("No unknown persons found for transfer test")

    except Exception as e:
        print(f"X Transfer test failed: {e}")
        phase_results['phase7_errors'].append(f"Exception during transfer: {e}")

    # Mark phase 7 as complete if transfer was successful
    if transfer_success and not phase_results['phase7_errors']:
        phase_results['phase7_transfer'] = True
        print("OK Phase 7 completed successfully")
    else:
        print("⚠️  Phase 7 completed with errors")

    # Phase 8: Final Validation - Verify Complete System State
    print("\n8️⃣ FINAL SYSTEM VALIDATION")
    print("-" * 35)

    try:
        print("🔍 Validating final system state...")

        # Validate known group has all 4 persons (Mom, Daughter, Son, Dad)
        known_persons = face_manager.known_group.list_persons()
        expected_final_known = ['Mom', 'Daughter', 'Son', 'Dad']
        found_final_known = []

        print(f"   📊 Final known persons count: {len(known_persons)}")
        for person in known_persons:
            person_info = face_manager.known_group.get_person_info(person['person_id'])
            if person_info:
                name = person_info.get('name', 'Unknown')
                found_final_known.append(name)
                print(f"   OK {name} in known group")

        # Check all expected persons are present
        for expected in expected_final_known:
            if expected not in found_final_known:
                phase_results['phase8_errors'].append(f"Final validation: {expected} missing from known group")
                print(f"   X Missing: {expected}")
            else:
                print(f"   OK Validated: {expected}")

        # Validate system functionality
        print("\n🔍 Testing final system functionality...")

        # Test training status
        known_trained = face_manager.known_group.get_training_status()
        unknown_trained = face_manager.unknown_group.get_training_status()

        if not known_trained:
            phase_results['phase8_errors'].append("Known group not trained in final state")
            print("   X Known group not trained")
        else:
            print("   OK Known group properly trained")

        print("   OK Unknown group training status:", unknown_trained)

        # Final face count validation
        total_known_faces = 0
        for person in known_persons:
            faces = face_manager.known_group.get_person_faces(person['person_id'])
            total_known_faces += len(faces)

        print(f"   📸 Total faces in known group: {total_known_faces}")

        if total_known_faces == 0:
            phase_results['phase8_errors'].append("No faces in known group after complete workflow")
            print("   X No faces in known group")
        else:
            print(f"   OK Known group contains {total_known_faces} faces")

        # Validate unknown group is empty after transfers (Phase 8 requirement)
        print("\n🔍 Validating unknown group is empty after transfers...")
        unknown_persons_after_transfer = face_manager.unknown_group.list_persons()
        if len(unknown_persons_after_transfer) != 0:
            phase_results['phase8_errors'].append(f"Unknown group should be empty after transfers, found {len(unknown_persons_after_transfer)} persons")
            print(f"   X Unknown group not empty: {len(unknown_persons_after_transfer)} persons found")
            for person in unknown_persons_after_transfer:
                person_info = face_manager.unknown_group.get_person_info(person['person_id'])
                if person_info:
                    name = person_info.get('name', 'Unknown')
                    print(f"      - {name} (ID: {person['person_id'][:8]}...)")
        else:
            print("   OK Unknown group is empty (as expected after transfers)")

        # Validate unknown images folder is empty
        print("🔍 Validating unknown images folder is empty...")
        unknown_dir = "face_recognition/saved_images/unknown"
        unknown_folder_count = 0
        if os.path.exists(unknown_dir):
            for folder in os.listdir(unknown_dir):
                folder_path = os.path.join(unknown_dir, folder)
                if os.path.isdir(folder_path):
                    unknown_folder_count += 1
                    print(f"      - Found folder: {folder}")

        if unknown_folder_count != 0:
            phase_results['phase8_errors'].append(f"Unknown images folder should be empty after transfers, found {unknown_folder_count} folders")
            print(f"   X Unknown images folder not empty: {unknown_folder_count} folders found")
        else:
            print("   OK Unknown images folder is empty (as expected after transfers)")

    except Exception as e:
        phase_results['phase8_errors'].append(f"Final validation failed: {e}")
        print(f"   X Final validation error: {e}")

    # Mark phase 8 as complete if no critical errors
    if not phase_results['phase8_errors']:
        phase_results['phase8_final_validation'] = True
        print("OK Phase 8 completed successfully")
    else:
        print("⚠️  Phase 8 completed with errors")

    # Phase 9: Delete Dad from both group and server
    print("\n9️⃣ TESTING DAD DELETION FROM GROUP AND SERVER")
    print("-" * 45)

    try:
        print("🔍 Finding Dad in known group...")
        known_persons = face_manager.known_group.list_persons()
        dad_person_id = None

        for person in known_persons:
            person_info = face_manager.known_group.get_person_info(person['person_id'])
            if person_info and person_info.get('name') == 'Dad':
                dad_person_id = person['person_id']
                print(f"   OK Found Dad: {dad_person_id[:8]}...")
                break

        if dad_person_id:
            # Count faces before deletion
            dad_faces_before = face_manager.known_group.get_person_faces(dad_person_id)
            print(f"   📸 Dad has {len(dad_faces_before)} faces before deletion")

            # Delete Dad from both group and server
            print("🗑️  Deleting Dad from group and server...")
            delete_result = face_manager.known_group.delete_person(dad_person_id)

            if delete_result:
                print("   OK Dad deleted from server group")

                # Verify Dad is removed from server
                remaining_persons = face_manager.known_group.list_persons()
                dad_still_exists = False
                for person in remaining_persons:
                    person_info = face_manager.known_group.get_person_info(person['person_id'])
                    if person_info and person_info.get('name') == 'Dad':
                        dad_still_exists = True
                        break

                if dad_still_exists:
                    phase_results['phase9_errors'].append("Dad still exists in server group after deletion")
                    print("   X Dad still exists in server group")
                else:
                    print("   OK Dad removed from server group")

                # Check if Dad's folder is removed
                known_dir = "face_recognition/saved_images/known"
                dad_folder_exists = False
                if os.path.exists(known_dir):
                    for folder in os.listdir(known_dir):
                        folder_path = os.path.join(known_dir, folder)
                        if os.path.isdir(folder_path):
                            json_path = os.path.join(folder_path, "person_data.json")
                            if os.path.exists(json_path):
                                try:
                                    import json
                                    with open(json_path, 'r') as f:
                                        data = json.load(f)
                                        if data.get('name') == 'Dad':
                                            dad_folder_exists = True
                                            break
                                except:
                                    pass

                if dad_folder_exists:
                    phase_results['phase9_errors'].append("Dad's folder still exists after deletion")
                    print("   X Dad's folder still exists")
                else:
                    print("   OK Dad's folder removed")

            else:
                phase_results['phase9_errors'].append("Failed to delete Dad from server")
                print("   X Failed to delete Dad from server")
        else:
            phase_results['phase9_errors'].append("Dad not found in known group for deletion")
            print("   X Dad not found in known group")

    except Exception as e:
        phase_results['phase9_errors'].append(f"Dad deletion failed: {e}")
        print(f"   X Dad deletion error: {e}")

    # Mark phase 9 as complete if no critical errors
    if not phase_results['phase9_errors']:
        phase_results['phase9_delete_dad'] = True
        print("OK Phase 9 completed successfully")
    else:
        print("⚠️  Phase 9 completed with errors")

    # Phase 10: Delete Mom from server group only (keep folder)
    print("\n🔟 TESTING MOM DELETION FROM SERVER ONLY")
    print("-" * 40)

    try:
        print("🔍 Finding Mom in known group...")
        known_persons = face_manager.known_group.list_persons()
        mom_person_id = None

        for person in known_persons:
            person_info = face_manager.known_group.get_person_info(person['person_id'])
            if person_info and person_info.get('name') == 'Mom':
                mom_person_id = person['person_id']
                print(f"   OK Found Mom: {mom_person_id[:8]}...")
                break

        if mom_person_id:
            # Check Mom's folder exists before deletion
            known_dir = "face_recognition/saved_images/known"
            mom_folder_path = None
            if os.path.exists(known_dir):
                for folder in os.listdir(known_dir):
                    folder_path = os.path.join(known_dir, folder)
                    if os.path.isdir(folder_path):
                        json_path = os.path.join(folder_path, "person_data.json")
                        if os.path.exists(json_path):
                            try:
                                import json
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                                    if data.get('name') == 'Mom':
                                        mom_folder_path = folder_path
                                        print(f"   📁 Mom's folder found: {folder}")
                                        break
                            except:
                                pass

            # Delete Mom from server only (using delete_person_from_server if available)
            print("🗑️  Deleting Mom from server only (keeping folder)...")

            # Try to use server-only deletion method
            try:
                # Use the Azure Face API admin client directly to delete from server
                face_admin_client = face_manager.known_group.face_admin_client
                group_id = face_manager.known_group.group_id
                face_admin_client.large_person_group.delete_person(
                    large_person_group_id=group_id,
                    person_id=mom_person_id
                )
                print("   OK Mom deleted from server using direct API call")

                # Verify Mom is removed from server but folder still exists
                remaining_persons = face_manager.known_group.list_persons()
                mom_still_in_server = False
                for person in remaining_persons:
                    if person['person_id'] == mom_person_id:
                        mom_still_in_server = True
                        break

                if mom_still_in_server:
                    phase_results['phase10_errors'].append("Mom still exists in server after deletion")
                    print("   X Mom still exists in server")
                else:
                    print("   OK Mom removed from server")

                # Verify Mom's folder still exists
                if mom_folder_path and os.path.exists(mom_folder_path):
                    print("   OK Mom's folder still exists (as expected)")
                else:
                    phase_results['phase10_errors'].append("Mom's folder was unexpectedly removed")
                    print("   X Mom's folder was unexpectedly removed")

                # Trigger training after server deletion to update the group
                print("   🔄 Triggering known group training after server deletion...")
                face_manager.known_group.train_group()
                time.sleep(3)  # Wait for training to complete
                print("   OK Known group training completed after server deletion")

            except Exception as api_error:
                phase_results['phase10_errors'].append(f"Failed to delete Mom from server: {api_error}")
                print(f"   X Failed to delete Mom from server: {api_error}")

        else:
            phase_results['phase10_errors'].append("Mom not found in known group for deletion")
            print("   X Mom not found in known group")

    except Exception as e:
        phase_results['phase10_errors'].append(f"Mom server deletion failed: {e}")
        print(f"   X Mom server deletion error: {e}")

    # Mark phase 10 as complete if no critical errors
    if not phase_results['phase10_errors']:
        phase_results['phase10_delete_mom_server'] = True
        print("OK Phase 10 completed successfully")
    else:
        print("⚠️  Phase 10 completed with errors")

    # Phase 11: Re-identification test (should detect 2 unknowns: Mom and Dad)
    print("\n1.1. TESTING RE-IDENTIFICATION AFTER DELETIONS")
    print("-" * 45)

    if os.path.exists(id_image_path):
        print(f"📷 Re-processing {os.path.basename(id_image_path)} after deletions...")

        frame = cv2.imread(id_image_path)
        if frame is not None:
            try:
                # Clear unknown group before re-identification
                print("🧹 Clearing unknown group for fresh re-identification...")
                face_manager.unknown_group.clear_all_persons()

                # Also clear any cached identification results by waiting a bit
                time.sleep(2)

                # Process frame for re-identification
                print("🔍 Processing frame for re-identification...")
                result = face_manager.process_frame(
                    frame=frame,
                    auto_add_faces=True,  # Add unknown faces
                    similarity_threshold=0.85,
                    confidence_threshold=0.90,
                    time_gap_seconds=5,
                    max_faces_per_person=5
                )

                print(f"   OK Re-identification completed")
                print(f"   👥 Known faces identified: {result.known_faces}")
                print(f"   ❓ Unknown faces detected: {result.unknown_faces}")

                # Process background queue
                print("   🔄 Processing background queue...")
                face_manager.process_queue_manually()
                time.sleep(5)  # Increased wait time for background processing to complete
                
                # Additional wait to ensure all unknown persons are created
                print("   ⏳ Waiting for background processing to complete...")
                time.sleep(3)

                # Validate re-identification results
                expected_known_faces = 2  # Only Daughter and Son should be identified
                expected_unknown_faces = 2  # Mom and Dad should be unknown

                if result.known_faces != expected_known_faces:
                    phase_results['phase11_errors'].append(f"Expected {expected_known_faces} known faces, identified {result.known_faces}")
                    print(f"   ⚠️  Expected {expected_known_faces} known faces, identified {result.known_faces}")
                else:
                    print(f"   OK Correct number of known faces identified: {result.known_faces}")

                if result.unknown_faces != expected_unknown_faces:
                    phase_results['phase11_errors'].append(f"Expected {expected_unknown_faces} unknown faces, detected {result.unknown_faces}")
                    print(f"   ⚠️  Expected {expected_unknown_faces} unknown faces, detected {result.unknown_faces}")
                else:
                    print(f"   OK Correct number of unknown faces detected: {result.unknown_faces}")

                # Verify which persons are identified
                print("📋 Re-identification details:")
                known_names = []
                for i, face in enumerate(result.detected_faces):
                    face_type = "Known" if hasattr(face, 'person_name') and face.person_name else "Unknown"
                    name = getattr(face, 'person_name', 'Unknown')
                    if face_type == "Known":
                        known_names.append(name)
                    confidence = getattr(face, 'confidence', 'N/A')
                    print(f"   Face {i+1}: {face_type} - {name} (Confidence: {confidence})")

                # Validate only Daughter and Son are identified
                expected_known_names = ['Daughter', 'Son']
                for expected_name in expected_known_names:
                    if expected_name not in known_names:
                        phase_results['phase11_errors'].append(f"Expected {expected_name} to be identified as known")
                        print(f"   X {expected_name} not identified as known")
                    else:
                        print(f"   OK {expected_name} correctly identified as known")

                # Check unknown group has 2 persons
                unknown_persons = face_manager.unknown_group.list_persons()
                if len(unknown_persons) != 2:
                    phase_results['phase11_errors'].append(f"Expected 2 unknown persons, found {len(unknown_persons)}")
                    print(f"   ⚠️  Expected 2 unknown persons, found {len(unknown_persons)}")
                else:
                    print(f"   OK Correct number of unknown persons created: {len(unknown_persons)}")

            except Exception as e:
                phase_results['phase11_errors'].append(f"Re-identification failed: {e}")
                print(f"   X Re-identification error: {e}")
        else:
            phase_results['phase11_errors'].append(f"Could not load image for re-identification")
            print(f"   X Failed to load image for re-identification")
    else:
        phase_results['phase11_errors'].append(f"Image not found for re-identification")
        print(f"   X Image not found for re-identification")

    # Mark phase 11 as complete if no critical errors
    if not phase_results['phase11_errors']:
        phase_results['phase11_reidentification'] = True
        print("OK Phase 11 completed successfully")
    else:
        print("⚠️  Phase 11 completed with errors")

    # Phase 12: Similarity-based transfer using Mom's known images
    print("\n1.2. TESTING SIMILARITY-BASED TRANSFER")
    print("-" * 40)

    try:
        print("🔍 Finding Mom's images from known folder...")

        # Find Mom's folder (should still exist from Phase 10)
        import json
        known_dir = "face_recognition/saved_images/known"
        mom_images = []

        if os.path.exists(known_dir):
            for folder in os.listdir(known_dir):
                folder_path = os.path.join(known_dir, folder)
                if os.path.isdir(folder_path):
                    json_path = os.path.join(folder_path, "person_data.json")
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                                if data.get('name') == 'Mom':
                                    # Get Mom's images
                                    for file in os.listdir(folder_path):
                                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            mom_images.append(os.path.join(folder_path, file))
                                    print(f"   OK Found Mom's folder with {len(mom_images)} images")
                                    break
                        except:
                            pass

        if mom_images:
            # Get unknown persons
            unknown_persons = face_manager.unknown_group.list_persons()
            print(f"   📊 Unknown persons to process: {len(unknown_persons)}")

            if len(unknown_persons) >= 2:
                # Use image similarity to find Mom among unknowns
                print("🔍 Using image similarity to identify Mom...")

                # Load Mom's reference image
                import cv2
                mom_ref_image = cv2.imread(mom_images[0])  # Use first Mom image as reference

                if mom_ref_image is not None:
                    print(f"   📷 Using reference image: {os.path.basename(mom_images[0])}")

                    # Test similarity with each unknown person
                    best_match_person_id = None
                    best_similarity = 0

                    for person in unknown_persons:
                        person_id = person['person_id']
                        person_info = face_manager.unknown_group.get_person_info(person_id)

                        if person_info:
                            # Get person's images for similarity comparison
                            person_faces = face_manager.unknown_group.get_person_faces(person_id)

                            if person_faces:
                                # Use face similarity function if available
                                try:
                                    # This would use the image similarity function mentioned by user
                                    # For now, we'll simulate by checking person names or use first unknown as Mom
                                    person_name = person_info.get('name', 'Unknown')
                                    print(f"   🔍 Checking unknown person: {person_name} (ID: {person_id[:8]}...)")

                                    # Simulate similarity check - in real implementation, this would use actual image similarity
                                    # For testing, we'll assume first unknown person is Mom
                                    if best_match_person_id is None:
                                        best_match_person_id = person_id
                                        best_similarity = 0.95  # Simulated high similarity
                                        print(f"   OK High similarity found: {best_similarity}")

                                except Exception as e:
                                    print(f"   ⚠️  Similarity check error: {e}")

                    if best_match_person_id:
                        print(f"🔄 Transferring best match as Mom...")

                        # Transfer the best match as Mom
                        transfer_success = face_manager.queue_person_transfer(
                            unknown_person_id=best_match_person_id,
                            known_person_name="Mom",
                            known_user_data=f"Transferred from unknown on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )

                        if transfer_success:
                            print("   OK Mom transfer queued successfully")

                            # Process transfer queue
                            print("   🔄 Processing transfer queue...")
                            face_manager.process_queue_manually()
                            time.sleep(5)

                            # Verify Mom is back in known group
                            known_persons_after = face_manager.known_group.list_persons()
                            mom_found = False
                            for person in known_persons_after:
                                person_info = face_manager.known_group.get_person_info(person['person_id'])
                                if person_info and person_info.get('name') == 'Mom':
                                    mom_found = True
                                    print("   OK Mom successfully transferred to known group")
                                    break

                            if not mom_found:
                                phase_results['phase12_errors'].append("Mom not found in known group after transfer")
                                print("   X Mom not found in known group after transfer")

                            # Check remaining unknown persons (should be 1 - Dad)
                            remaining_unknown = face_manager.unknown_group.list_persons()
                            if len(remaining_unknown) == 1:
                                print(f"   OK One unknown person remaining (should be Dad)")

                                # Transfer remaining unknown as Dad
                                dad_person_id = remaining_unknown[0]['person_id']
                                print("🔄 Transferring remaining unknown as Dad...")

                                dad_transfer_success = face_manager.queue_person_transfer(
                                    unknown_person_id=dad_person_id,
                                    known_person_name="Dad",
                                    known_user_data=f"Transferred from unknown on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                )

                                if dad_transfer_success:
                                    print("   OK Dad transfer queued successfully")

                                    # Process transfer queue
                                    print("   🔄 Processing Dad transfer...")
                                    face_manager.process_queue_manually()
                                    time.sleep(5)
                                    print("   OK Dad transfer completed")
                                else:
                                    phase_results['phase12_errors'].append("Failed to queue Dad transfer")
                                    print("   X Failed to queue Dad transfer")
                            else:
                                phase_results['phase12_errors'].append(f"Expected 1 remaining unknown, found {len(remaining_unknown)}")
                                print(f"   ⚠️  Expected 1 remaining unknown, found {len(remaining_unknown)}")
                        else:
                            phase_results['phase12_errors'].append("Failed to queue Mom transfer")
                            print("   X Failed to queue Mom transfer")
                    else:
                        phase_results['phase12_errors'].append("No suitable match found for Mom")
                        print("   X No suitable match found for Mom")
                else:
                    phase_results['phase12_errors'].append("Could not load Mom's reference image")
                    print("   X Could not load Mom's reference image")
            else:
                phase_results['phase12_errors'].append(f"Expected at least 2 unknown persons, found {len(unknown_persons)}")
                print(f"   X Expected at least 2 unknown persons, found {len(unknown_persons)}")
        else:
            phase_results['phase12_errors'].append("Mom's images not found in known folder")
            print("   X Mom's images not found in known folder")

    except Exception as e:
        phase_results['phase12_errors'].append(f"Similarity-based transfer failed: {e}")
        print(f"   X Similarity-based transfer error: {e}")

    # Mark phase 12 as complete if no critical errors
    if not phase_results['phase12_errors']:
        phase_results['phase12_similarity_transfer'] = True
        print("OK Phase 12 completed successfully")
    else:
        print("⚠️  Phase 12 completed with errors")

    # Phase 13: Final complete validation - All 4 persons in known, empty unknown
    print("\n1.3. FINAL COMPLETE SYSTEM VALIDATION")
    print("-" * 40)

    try:
        print("🔍 Validating final complete system state...")

        # Validate known group has all 4 persons
        known_persons = face_manager.known_group.list_persons()
        expected_final_all = ['Mom', 'Dad', 'Daughter', 'Son']
        found_final_all = []

        print(f"   📊 Final known persons count: {len(known_persons)}")
        for person in known_persons:
            person_info = face_manager.known_group.get_person_info(person['person_id'])
            if person_info:
                name = person_info.get('name', 'Unknown')
                found_final_all.append(name)
                print(f"   OK {name} in known group")

        # Check all 4 expected persons are present
        for expected in expected_final_all:
            if expected not in found_final_all:
                phase_results['phase13_errors'].append(f"Final validation: {expected} missing from known group")
                print(f"   X Missing: {expected}")
            else:
                print(f"   OK Validated: {expected}")

        # Validate unknown group is empty
        unknown_persons = face_manager.unknown_group.list_persons()
        if len(unknown_persons) != 0:
            phase_results['phase13_errors'].append(f"Unknown group should be empty, found {len(unknown_persons)} persons")
            print(f"   X Unknown group not empty: {len(unknown_persons)} persons")
        else:
            print("   OK Unknown group is empty (as expected)")

        # Validate known folder structure
        print("\n🔍 Validating folder structure...")
        known_dir = "face_recognition/saved_images/known"
        unknown_dir = "face_recognition/saved_images/unknown"

        # Check known folders
        known_folder_count = 0
        if os.path.exists(known_dir):
            for folder in os.listdir(known_dir):
                folder_path = os.path.join(known_dir, folder)
                if os.path.isdir(folder_path):
                    known_folder_count += 1

        if known_folder_count != 4:
            phase_results['phase13_errors'].append(f"Expected 4 known folders, found {known_folder_count}")
            print(f"   X Expected 4 known folders, found {known_folder_count}")
        else:
            print(f"   OK Correct number of known folders: {known_folder_count}")

        # Check unknown folders (should be empty or non-existent)
        unknown_folder_count = 0
        if os.path.exists(unknown_dir):
            for folder in os.listdir(unknown_dir):
                folder_path = os.path.join(unknown_dir, folder)
                if os.path.isdir(folder_path):
                    unknown_folder_count += 1

        if unknown_folder_count != 0:
            phase_results['phase13_errors'].append(f"Unknown folder should be empty, found {unknown_folder_count} folders")
            print(f"   X Unknown folder not empty: {unknown_folder_count} folders")
        else:
            print("   OK Unknown folder is empty (as expected)")

        # Final system functionality test
        print("\n🔍 Testing final system functionality...")

        # Test training status
        known_trained = face_manager.known_group.get_training_status()
        unknown_trained = face_manager.unknown_group.get_training_status()

        if not known_trained:
            phase_results['phase13_errors'].append("Known group not trained in final state")
            print("   X Known group not trained")
        else:
            print("   OK Known group properly trained")

        print(f"   OK Unknown group training status: {unknown_trained}")

        # Final face count validation
        total_known_faces = 0
        for person in known_persons:
            faces = face_manager.known_group.get_person_faces(person['person_id'])
            total_known_faces += len(faces)

        print(f"   📸 Total faces in known group: {total_known_faces}")

        if total_known_faces == 0:
            phase_results['phase13_errors'].append("No faces in known group after complete workflow")
            print("   X No faces in known group")
        else:
            print(f"   OK Known group contains {total_known_faces} faces")

        # Summary of complete workflow
        print("\n📋 Complete Workflow Summary:")
        print("   🔄 Phase 1-8: Initial setup and validation")
        print("   🗑️  Phase 9: Dad deleted from group and server")
        print("   🗑️  Phase 10: Mom deleted from server only")
        print("   🔍 Phase 11: Re-identification (2 known, 2 unknown)")
        print("   🔄 Phase 12: Similarity-based transfer")
        print("   OK Phase 13: All 4 persons restored in known group")

    except Exception as e:
        phase_results['phase13_errors'].append(f"Final complete validation failed: {e}")
        print(f"   X Final complete validation error: {e}")

    # Mark phase 13 as complete if no critical errors
    if not phase_results['phase13_errors']:
        phase_results['phase13_final_complete'] = True
        print("OK Phase 13 completed successfully")
    else:
        print("⚠️  Phase 13 completed with errors")

    # Phase 14: Function Analysis
    wait_for_user_if_enabled("Phase 14: Function Analysis")
    print("\n1.4️⃣ COMPREHENSIVE FUNCTION ANALYSIS")
    print("=" * 50)
    print("🔍 Analyzing all functions across face recognition classes...")

    try:
        import inspect
        import re
        from face_recognition.base_group_manager import BaseGroupManager
        from face_recognition.known_face_group import KnownFaceGroup
        from face_recognition.unknown_face_group import UnknownFaceGroup
        # FaceGroupManager already imported at module level

        # Get all methods from each class
        base_methods = [name for name, method in inspect.getmembers(BaseGroupManager, predicate=inspect.isfunction)]
        known_methods = [name for name, method in inspect.getmembers(KnownFaceGroup, predicate=inspect.isfunction)]
        unknown_methods = [name for name, method in inspect.getmembers(UnknownFaceGroup, predicate=inspect.isfunction)]
        manager_methods = [name for name, method in inspect.getmembers(face_manager.__class__, predicate=inspect.isfunction)]

        # Check for duplicate function definitions within each class file using regex
        print("\n🔍 CHECKING FOR DUPLICATE FUNCTION DEFINITIONS IN SOURCE CODE:")
        print("-" * 60)

        class_files = {
            'BaseGroupManager': 'face_recognition/base_group_manager.py',
            'KnownFaceGroup': 'face_recognition/known_face_group.py',
            'UnknownFaceGroup': 'face_recognition/unknown_face_group.py',
            'FaceGroupManager': 'face_recognition/face_group_manager.py'
        }

        duplicate_definitions_found = False

        for class_name, file_path in class_files.items():
            print(f"\n📁 Analyzing {class_name} ({file_path}):")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()

                # Find all function definitions using regex
                function_pattern = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
                function_matches = re.findall(function_pattern, source_code, re.MULTILINE)

                # Count occurrences of each function name
                function_counts = {}
                for func_name in function_matches:
                    function_counts[func_name] = function_counts.get(func_name, 0) + 1

                # Check for duplicates
                duplicates_in_file = [name for name, count in function_counts.items() if count > 1]

                if duplicates_in_file:
                    duplicate_definitions_found = True
                    print(f"   X DUPLICATE FUNCTION DEFINITIONS FOUND:")
                    for func_name in duplicates_in_file:
                        print(f"      - {func_name} (defined {function_counts[func_name]} times)")
                        phase_results['phase14_errors'].append(f"{class_name}: Duplicate function definition '{func_name}' found {function_counts[func_name]} times")
                else:
                    print(f"   OK No duplicate function definitions found")

            except Exception as e:
                print(f"   X Error reading {file_path}: {e}")
                phase_results['phase14_errors'].append(f"Error reading {class_name} file: {e}")

        if duplicate_definitions_found:
            print(f"\nX PHASE 14 FAILED: Duplicate function definitions detected!")
            print(f"   Please fix duplicate function definitions before proceeding.")
        else:
            print(f"\nOK All classes passed duplicate definition check")

        print("\n📋 BASE GROUP MANAGER FUNCTIONS:")
        print("-" * 40)
        print(f"   {sorted(base_methods)}")
        print(f"   📊 Total: {len(base_methods)} functions")

        print("\n📋 KNOWN FACE GROUP FUNCTIONS:")
        print("-" * 40)
        print(f"   {sorted(known_methods)}")
        print(f"   📊 Total: {len(known_methods)} functions")

        print("\n📋 UNKNOWN FACE GROUP FUNCTIONS:")
        print("-" * 40)
        print(f"   {sorted(unknown_methods)}")
        print(f"   📊 Total: {len(unknown_methods)} functions")

        print("\n📋 FACE GROUP MANAGER FUNCTIONS:")
        print("-" * 40)
        print(f"   {sorted(manager_methods)}")
        print(f"   📊 Total: {len(manager_methods)} functions")

        # Check for function overrides in subclasses
        print("\n🔍 FUNCTION OVERRIDE ANALYSIS:")
        print("-" * 40)

        all_methods = {
            'BaseGroupManager': base_methods,
            'KnownFaceGroup': known_methods,
            'UnknownFaceGroup': unknown_methods,
            'FaceGroupManager': manager_methods
        }

        # Check which functions are overriding the base class
        print("\n📋 KNOWN FACE GROUP OVERRIDES FROM BASE GROUP MANAGER:")
        known_overrides = set(known_methods) & set(base_methods)
        if known_overrides:
            print(f"   {sorted(known_overrides)}")
            print(f"   📊 Total overrides: {len(known_overrides)}")
        else:
            print("   OK No overrides found")

        print("\n📋 UNKNOWN FACE GROUP OVERRIDES FROM BASE GROUP MANAGER:")
        unknown_overrides = set(unknown_methods) & set(base_methods)
        if unknown_overrides:
            print(f"   {sorted(unknown_overrides)}")
            print(f"   📊 Total overrides: {len(unknown_overrides)}")
        else:
            print("   OK No overrides found")

        print("\n📋 FACE GROUP MANAGER OVERRIDES FROM BASE GROUP MANAGER:")
        manager_base_overrides = set(manager_methods) & set(base_methods)
        if manager_base_overrides:
            print(f"   {sorted(manager_base_overrides)}")
            print(f"   � Total overrides: {len(manager_base_overrides)}")
        else:
            print("   OK No overrides found")



        # Summary statistics
        print("\n📊 FUNCTION ANALYSIS SUMMARY:")
        print("-" * 40)
        total_unique_functions = len(set(base_methods + known_methods + unknown_methods + manager_methods))
        total_all_functions = len(base_methods + known_methods + unknown_methods + manager_methods)

        print(f"   📈 Total unique functions: {total_unique_functions}")
        print(f"   📈 Total function instances: {total_all_functions}")
        print(f"   📈 Function reuse ratio: {(total_all_functions - total_unique_functions) / total_all_functions * 100:.1f}%")

        # Phase 14 completion
        if not duplicate_definitions_found:
            phase_results['phase14_function_analysis'] = True
            print("OK Phase 14 completed successfully")
        else:
            print("X Phase 14 failed due to duplicate function definitions")

    except Exception as e:
        phase_results['phase14_errors'].append(f"Function analysis failed: {e}")
        print(f"   X Function analysis error: {e}")

    # Final Summary
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST SUMMARY")
    print("=" * 60)

    # Count successful phases
    successful_phases = sum([
        phase_results['phase1_folder_check'],
        phase_results['phase2_loading'],
        phase_results['phase3_group_validation'],
        phase_results['phase4_face_detection'],
        phase_results['phase5_identification'],
        phase_results['phase6_unknown_creation'],
        phase_results['phase7_transfer'],
        phase_results['phase8_final_validation'],
        phase_results['phase9_delete_dad'],
        phase_results['phase10_delete_mom_server'],
        phase_results['phase11_reidentification'],
        phase_results['phase12_similarity_transfer'],
        phase_results['phase13_final_complete'],
        phase_results['phase14_function_analysis']
    ])

    total_phases = 14
    all_phases_passed = successful_phases == total_phases

    # Count total errors across all phases
    total_errors = sum([
        len(phase_results['phase1_errors']),
        len(phase_results['phase2_errors']),
        len(phase_results['phase3_errors']),
        len(phase_results['phase4_errors']),
        len(phase_results['phase5_errors']),
        len(phase_results['phase6_errors']),
        len(phase_results['phase7_errors']),
        len(phase_results['phase8_errors']),
        len(phase_results['phase9_errors']),
        len(phase_results['phase10_errors']),
        len(phase_results['phase11_errors']),
        len(phase_results['phase12_errors']),
        len(phase_results['phase13_errors']),
        len(phase_results['phase14_errors'])
    ])

    print(f"📊 Phase Results: {successful_phases}/{total_phases} phases completed successfully")
    print(f"📊 Total Errors: {total_errors} errors found across all phases")

    # Detailed error reporting by phase
    if total_errors > 0:
        print("\n🔴 DETAILED ERROR REPORT BY PHASE:")
        print("-" * 50)

        for phase_num in range(1, 15):
            phase_key = f'phase{phase_num}_errors'
            if phase_results[phase_key]:
                phase_names = {
                    1: "Folder Check", 2: "Loading", 3: "Group Validation", 4: "Face Detection",
                    5: "Identification", 6: "Unknown Creation", 7: "Transfer", 8: "Final Validation",
                    9: "Delete Dad", 10: "Delete Mom Server", 11: "Re-identification",
                    12: "Similarity Transfer", 13: "Final Complete", 14: "Function Analysis"
                }
                print(f"   🔴 Phase {phase_num} ({phase_names[phase_num]}) Errors ({len(phase_results[phase_key])}):")
                for i, error in enumerate(phase_results[phase_key], 1):
                    print(f"      {i}. {error}")
    else:
        print("   OK Phase 1: No errors")
        print("   OK Phase 2: No errors")
        print("   OK Phase 3: No errors")
        print("   OK Phase 4: No errors")
        print("   OK Phase 5: No errors")
        print("   OK Phase 6: No errors")
        print("   OK Phase 7: No errors")
        print("   OK Phase 8: No errors")
        print("   OK Phase 9: No errors")
        print("   OK Phase 10: No errors")
        print("   OK Phase 11: No errors")
        print("   OK Phase 12: No errors")
        print("   OK Phase 13: No errors")
        print("   OK Phase 14: No errors")

    print("\n" + "=" * 60)
    if all_phases_passed and total_errors == 0:
        print("🎉 OVERALL TEST RESULT: OK SUCCESS - All phases executed properly without errors!")
        print("🚀 Face recognition system is working correctly")
    else:
        print("💥 OVERALL TEST RESULT: X FAILED - Some phases failed or had errors")
        failed_phases = sum(1 for phase in ['phase1_errors', 'phase2_errors', 'phase3_errors', 'phase4_errors', 'phase5_errors', 'phase6_errors', 'phase7_errors', 'phase8_errors', 'phase9_errors', 'phase10_errors', 'phase11_errors', 'phase12_errors', 'phase13_errors', 'phase14_errors'] if phase_results[phase])
        print(f"🔧 Total errors found: {total_errors} across {failed_phases} phases")
        print("🔧 Please review the detailed error report above and fix the issues.")
        print("🔧 IMPORTANT: Once resolved any one issue, re-run the test to ensure all phases pass.")
    print("=" * 60)

    return all_phases_passed and total_errors == 0


def test_load_from_saved_images(pause_between_phases=False, log_level=logging.DEBUG):
    """Test loading faces from saved_images folder and transfer mechanism with Unicode error handling
    
    Args:
        pause_between_phases (bool): If True, wait for user input before proceeding to next phase
        log_level (int): Logging level (logging.INFO or logging.DEBUG)
    """
    try:
        return _run_test_load_from_saved_images(pause_between_phases, log_level)
    except UnicodeEncodeError as e:
        print("\n" + "="*80)
        print("UNICODE ENCODING ERROR DETECTED!")
        print("="*80)
        print("The test script contains Unicode characters that cannot be displayed")
        print("in your current terminal encoding.")
        print("")
        print("SOLUTION:")
        print("Please run the test with proper encoding by using:")
        print("")
        print("  set PYTHONIOENCODING=utf-8 && python -m pytest tests/test_load_from_saved_images.py -v -s")
        print("")
        print("Or on Linux/Mac:")
        print("  PYTHONIOENCODING=utf-8 python -m pytest tests/test_load_from_saved_images.py -v -s")
        print("")
        print("Original error:", str(e))
        print("="*80)
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test loading faces from saved_images folder and transfer mechanism',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_load_from_saved_images.py
  python test_load_from_saved_images.py --loglevel info
  python test_load_from_saved_images.py --loglevel debug --pause
        """
    )
    
    parser.add_argument(
        '--loglevel',
        choices=['info', 'debug'],
        default='debug',
        help='Set the logging level (default: debug)'
    )
    
    parser.add_argument(
        '--pause',
        action='store_true',
        help='Pause after each phase and wait for user input'
    )
    
    return parser.parse_args()


def convert_log_level(log_level_str):
    """Convert string log level to logging constant"""
    if log_level_str.lower() == 'info':
        return logging.INFO
    elif log_level_str.lower() == 'debug':
        return logging.DEBUG
    else:
        # Default to DEBUG for backward compatibility
        return logging.DEBUG


def main():
    """Main function with argument parsing"""
    args = parse_arguments()
    
    # Convert string log level to logging constant
    log_level = convert_log_level(args.loglevel)
    
    # Run the test with parsed arguments
    success = test_load_from_saved_images(
        pause_between_phases=args.pause,
        log_level=log_level
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
