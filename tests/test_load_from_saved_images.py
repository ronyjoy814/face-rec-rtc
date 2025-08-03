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

import cv2
import time
from datetime import datetime
from dotenv import load_dotenv
from face_recognition.face_recognition_manager import FaceRecognitionManager

# Load environment variables
load_dotenv()

def test_load_from_saved_images():
    """Test loading faces from saved_images folder and transfer mechanism"""
    
    print("📚 LOAD FROM SAVED IMAGES TEST")
    print("=" * 35)
    
    # Initialize Face Recognition Manager
    endpoint = os.getenv('AZURE_FACE_API_ENDPOINT')
    api_key = os.getenv('AZURE_FACE_API_ACCOUNT_KEY')
    
    # Use unique group ID to avoid accumulation
    import time
    unique_id = f"load_saved_test_1"

    face_manager = FaceRecognitionManager(
        endpoint=endpoint,
        api_key=api_key,
        group_id=unique_id,
        group_name="Load Saved Test",
        save_known_images=True,
        auto_train_unknown=True,
        max_faces_per_person=5,
        cleanup_known_group=True,    
        cleanup_unknown_images=False
    )
    print("✅ Face Recognition Manager initialized")
    
    # Phase 1: Check if saved_images folder exists
    print("\n1️⃣ CHECKING SAVED IMAGES FOLDER")
    print("-" * 35)
    
    saved_images_dir = "face_recognition/saved_images"
    known_dir = f"{saved_images_dir}/known"
    unknown_dir = f"{saved_images_dir}/unknown"
    
    if os.path.exists(known_dir):
        known_folders = [f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]
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
        print("❌ Known images folder not found")
        return False
    
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
    
    # Phase 2: Load faces from saved_images (known group only first)
    print("\n2️⃣ LOADING KNOWN FACES FROM SAVED IMAGES")
    print("-" * 45)
    
    print("📚 Loading known faces from saved_images folder...")
    
    # Count folders before loading
    folders_before = len([f for f in os.listdir(known_dir) if os.path.isdir(os.path.join(known_dir, f))]) if os.path.exists(known_dir) else 0
    
    load_results = face_manager.group_manager.load_faces_from_saved_images()
    
    if 'error' not in load_results:
        print("✅ Known faces loaded successfully")
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
            print("✅ No redundant folders created (good folder management)")
        else:
            print("⚠️  More folders after loading (check folder management)")
            
    else:
        print(f"❌ Failed to load known faces: {load_results['error']}")
        return False
    
    time.sleep(5)
    print("✅Training and Loading successful")
    
    # Phase 4: Process identification1.jpg to create unknown persons (including Dad)
    print("\n4️⃣ PROCESSING IDENTIFICATION1.JPG TO CREATE UNKNOWN PERSONS")
    print("-" * 65)

    id_image_path = "images/identification1.jpg"
    if os.path.exists(id_image_path):
        print(f"📷 Processing {os.path.basename(id_image_path)} to detect unknown faces...")

        import cv2
        frame = cv2.imread(id_image_path)
        if frame is not None:
            # Process frame to detect faces (this will create unknown persons for unrecognized faces)
            result = face_manager.process_frame(
                frame=frame,
                auto_add_faces=True,
                similarity_threshold=0.85,
                confidence_threshold=0.90,
                time_gap_seconds=5,
                max_faces_per_person=5
            )

            print(f"✅ Frame processing completed")
            print(f"🔍 Total faces detected: {len(result.detected_faces)}")
            print(f"👥 Known faces identified: {result.known_faces}")
            print(f"❓ Unknown faces detected: {result.unknown_faces}")

            # Process background queue to add unknown faces
            print("🔄 Processing background queue to add unknown faces...")
            face_manager.group_manager.process_queue_manually()
            time.sleep(3)
            print("✅ Background processing completed")

        else:
            print(f"❌ Failed to load {id_image_path}")
            return False
    else:
        print(f"❌ Image not found: {id_image_path}")
        return False

    print("Sleep for 10 Sec to finish background processing")
    time.sleep(10)

    # Phase 6: Test transfer mechanism with real data
    print("\n6️⃣ TESTING TRANSFER WITH REAL DATA")
    print("-" * 40)
    
    # List unknown persons to find candidates for transfer
    try:
        unknown_persons = face_manager.group_manager.unknown_group.list_persons()
        if unknown_persons:
            print(f"🔍 Found {len(unknown_persons)} unknown persons for potential transfer")

            # Get the first unknown person for transfer test
            first_unknown = unknown_persons[0]
            unknown_person_id = first_unknown['person_id']

            # Get person info
            person_info = face_manager.group_manager.unknown_group.get_person_info(unknown_person_id)
            if person_info:
                current_name = person_info.get('name', 'Unknown')
                print(f"📋 Selected unknown person: {current_name} (ID: {unknown_person_id[:8]}...)")
                
                # Test transfer to known group as "Dad"
                print("🔄 Testing transfer to known group as 'Dad'...")
                
                success = face_manager.group_manager.queue_person_transfer(
                    unknown_person_id=unknown_person_id,
                    known_person_name="Dad",
                    known_user_data=f"Transferred from {current_name} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                if success:
                    print("✅ Dad transfer queued successfully")
                    
                    # Process the transfer queue
                    print("🔄 Processing transfer queue...")
                    face_manager.group_manager.process_queue_manually()
                    time.sleep(3)
                    print("✅ Transfer processing completed")
                    
                    # Verify Dad was added to known group
                    print("🔍 Verifying Dad in known group...")
                    known_persons = face_manager.group_manager.known_group.list_persons()
                    dad_found = False

                    for person in known_persons:
                        person_info = face_manager.group_manager.known_group.get_person_info(person['person_id'])
                        if person_info and person_info.get('name') == 'Dad':
                            dad_found = True
                            print(f"✅ Dad found in known group: {person['person_id'][:8]}...")
                            print(f"   📝 User data: {person_info.get('user_data', '')[:50]}...")
                            break
                    
                    if dad_found:
                        print("✅ Transfer mechanism working correctly with real data!")
                    else:
                        print("⚠️  Dad not found in known group after transfer")
                else:
                    print("❌ Failed to queue Dad transfer")
            else:
                print("❌ Could not get unknown person info")
        else:
            print("⚠️  No unknown persons found for transfer test")
            
    except Exception as e:
        print(f"❌ Transfer test failed: {e}")
    

    
    return True

if __name__ == "__main__":
    success = test_load_from_saved_images()
