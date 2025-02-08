import os
import json

def get_face_filenames():
    faces_dir = "./public/faces"
    billionaires_file = "./billionaires.json"
    face_mapping = {}
    
    # Load billionaires data
    with open(billionaires_file, 'r', encoding='utf-8') as f:
        billionaires = json.load(f)
    
    # Get all filenames in the faces directory if it exists
    existing_faces = set()
    if os.path.exists(faces_dir):
        existing_faces = set(os.listdir(faces_dir))
    
    # Check each billionaire against existing face images
    for k, v in billionaires.items():
        expected_filename = v["name"].lower().replace(" ", "-") + ".jpg"
        if expected_filename in existing_faces:
            face_mapping[expected_filename] = v
    
    # Write the mapping to a JSON file
    with open('public/face_filenames.json', 'w') as f:
        json.dump(face_mapping, f)

if __name__ == "__main__":
    get_face_filenames()
