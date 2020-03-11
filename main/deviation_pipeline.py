import json
import os
from scipy.spatial.distance import directed_hausdorff

class imageCollection:
    def __init__(self, img_id, landmarks, img_set):
        self.img_id = img_id
        self.landmarks = landmarks
        self.img_set = img_set

class deviationCollection:
    def __init__(self, warped_face_id, deviation_imgs):
        self.warped_face_id = warped_face_id
        self.deviation_imgs = deviation_imgs

def read_warped_faces(path):
    imageCollectionArray = []

    with open(path) as f:
        data = json.load(f)

    for entry in data:
        imageCollectionArray.append(imageCollection(entry, 
                                    data[entry]['landmarks'],
                                    data[entry]['images']))
    return imageCollectionArray

def read_face_collection(path):
    faceCollection = {}

    with open(path) as f:
        data = json.load(f)

    for entry in data:
        faceCollection[(entry['imgid'], entry['faceid'])] = entry['points']

    return faceCollection

def find_face(img, face_collection):
    if img in face_collection:
        return face_collection[img]
    else:
        return -1

def compute_deviations_to_warped_face(warped_collection, face_collection):
    deviation_collection = []

    for face in warped_collection:
        dev_to_warped = []
        img_visited = {}
        benchmark = face.landmarks

        for img in face.img_set:
            if img not in img_visited:
                img_visited[img] = 0
                landmarks = find_face((img, 0), face_collection)
            else:
                old_num = img_visited[img]
                img_visited[img] = old_num + 1
                landmarks = find_face((img, old_num + 1), face_collection)

            if landmarks != -1:
                distance = get_distance_between_vectors(benchmark, landmarks)
                dev_to_warped.append({"img_id": img, "face_id": img_visited[img], "deviation":distance})
        
        deviation_collection.append(deviationCollection(face.img_id, dev_to_warped))

    return deviation_collection

def get_distance_between_vectors(mean, vec):
    return directed_hausdorff(mean, vec)[0]

def sort_images(deviation_collection):
    for warped_face in deviation_collection:
        warped_face.deviation_imgs.sort(key=lambda x: x['deviation'])

def convert_to_serializable_format(deviation_collection):
    return { x.warped_face_id: x.deviation_imgs for x in deviation_collection }

def main():
    directory = "../data"

    face_collection = read_face_collection(directory+"/faces.json")
    
    for filename in os.listdir(directory):
        if filename.endswith(".json") and filename != "faces.json":
            warped_collection = read_warped_faces("../data/"+filename)
            
            deviation_collection = compute_deviations_to_warped_face(warped_collection, face_collection)
            sort_images(deviation_collection)

            with open('../output/output_'+filename, 'w') as outfile:
                json.dump(convert_to_serializable_format(deviation_collection), outfile)

if __name__ == "__main__":
    main()