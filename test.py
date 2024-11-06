import requests

# Prepare files
files = {
    
    'video1': open('/media/sigmind/URSTP_HDD1414/DeepStream-Yolo/001.mp4', 'rb'),
    'video2': open('/media/sigmind/URSTP_HDD1414/DeepStream-Yolo/044.mp4', 'rb'),
    'analytics_config': open('/media/sigmind/URSTP_HDD1414/DeepStream-Yolo/config_infer_primary.txt', 'rb'),
    'detector_config': open('/media/sigmind/URSTP_HDD1414/DeepStream-Yolo/config_infer_primary_damoyolo.txt', 'rb')
}

# Send request
response = requests.post('http://localhost:5000/upload', files=files)
print(response.json())