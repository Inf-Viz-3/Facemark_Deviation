import json

from numpy import sqrt, array
from scipy.spatial.distance import directed_hausdorff

def get_distance_between_vectors(mean, vec):
    return directed_hausdorff(mean, vec)[0]


def get_mean_points(json_dict, n):
    mean_points = []

    for i in range(n):
        c_x = []
        c_y = []
        
        for coordinate in json_dict.values():
            c_x.append(coordinate[i][0])
            c_y.append(coordinate[i][1])

        avg_c_x = round(sum(c_x)/len(c_x))
        avg_c_y = round(sum(c_y)/len(c_y))
    
        mean_points.append([avg_c_x, avg_c_y])

    return mean_points

def json_to_dict(json):
    d = {}
    for entry in json:
        d[entry['imgid']] = entry['points']
    
    return d

def read_data(path):
    with open(path) as f:
        data = json.load(f)
    
    lenght_points = len(data[0]['points'])

    return lenght_points, data


def main():
    n, json_data = read_data("data/faces.json")
    json_dict = json_to_dict(json_data)
    
    mean_points = get_mean_points(json_dict, n)

    for c_vector in json_dict.values():
        distance = get_distance_between_vectors(mean_points, c_vector)
        print(round(distance))


    
        





if __name__ == "__main__":
    main()