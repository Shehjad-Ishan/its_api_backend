from flask import Flask, request, jsonify
import os
import threading
from process import process_videos  # Assuming you have the process_videos function
import sys
from tqdm import tqdm
import time
from database_entry import insert_video_name, insert_analytics_file_path

sys.path.append('/home/sigmind/deepstream_sdk_v6.3.0_x86_64/opt/nvidia/deepstream/deepstream-6.3/sources/apps/deepstream_python_apps/apps/common')

app = Flask(__name__)

# Define a directory where the uploaded files will be saved temporarily
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def upload_videos_progress(videos):
    video_paths = []
    for video in tqdm(videos, desc="Uploading videos"):
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)  # Save file to the directory
        video_paths.append(video_path)  # Store the file path
        insert_video_name(video_path, 0)
    return video_paths

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Check for video files
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        # Get uploaded videos
        videos = request.files.getlist('video')
        
        # Optional: get additional configs
        detector_config = request.files.get('detector_config')
        analytics_config = request.files.get('analytics_config')

        # Upload videos with progress
        video_paths = upload_videos_progress(videos)
        
        # Save configs if provided
        config_paths = {}
        if detector_config:
            detector_path = os.path.join(UPLOAD_FOLDER, 'detector_config.txt')
            detector_config.save(detector_path)
            config_paths['detector'] = detector_path
        
        if analytics_config:
            input_name= input("what should be the name of the analytics config?")
            ana_name=input_name + '_' + 'analytics_config.txt'
            analytics_path = os.path.join(UPLOAD_FOLDER, f'{ana_name}')
            analytics_config.save(analytics_path)
            config_paths['analytics'] = analytics_path
            insert_analytics_file_path(analytics_path)

        # Start processing in a separate thread, passing video and config paths
        threading.Thread(
            target=process_videos,
            args=(video_paths, config_paths)  # Modify process_videos if needed to accept config paths
        ).start()
        
        return jsonify({
            'status': 'success',
            'message': f'Processing started for {len(videos)} videos with provided configs'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
