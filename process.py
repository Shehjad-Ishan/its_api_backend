import subprocess
import psutil
import logging
from typing import List, Dict
import os
from database_entry import update_complete_flag, show_all_config_files, get_config_path_by_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

selected = 0
ana_config_file= "/media/sigmind/URSTP_HDD1415/python_api/uploads/sabbir_analytics_config.txt"

def kill_process(process):
    """Kill a process and its children."""
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def run_command(video_path: str, config: Dict[str, str] ) -> None:
    """
    Run the video processing command.
    Modify this function to run your specific command.
    """
    global ana_config_file
    data_base_video_path= video_path
    parent_path= os.getcwd() + "/"
    video_path="file://" + parent_path + video_path
    detector_config_path = parent_path + config["detector"]
    #analytics_config_path = parent_path + config["analytics"]

    try:
        # Create the command with the video path
        cmd = [
            "python3",
            "-m",
            "urstp_OSD",  # Update with the correct script if needed
            video_path,
            "/media/sigmind/sig-data-pool/survey_video_extract",
            # analytics_config_path,
            ana_config_file,
            detector_config_path

        ]
        print("Process starting")
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Get output
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Process failed with error: {stderr.decode()}")
        else:
            logger.info(f"Process completed successfully for video: {video_path}")
            update_complete_flag(data_base_video_path)
            
    except Exception as e:
        logger.error(f"Error running command for video {video_path}: {str(e)}")

def process_videos(video_paths: List[str], config: Dict[str, str]) -> None:
    """
    Process multiple videos sequentially, ensuring only one runs at a time.
    """
    global ana_config_file
    global selected 
    if selected==0:
        show_all_config_files()
        config_id = int(input("Enter the configuration file ID: "))
        ana_config_file = get_config_path_by_id(config_id)
        ana_config_file = os.getcwd() + "/" + ana_config_file
        selected=1

    current_process = None
    try:
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path} with {ana_config_file}")
            
            # If there's a current process running, terminate it
            if current_process and current_process.poll() is None:
                logger.info("Terminating current process")
                kill_process(current_process)
            
            # Process the current video
            run_command(video_path, config)
            
    except Exception as e:
        logger.error(f"Error in process_videos: {str(e)}")
    
    finally:
        if current_process and current_process.poll() is None:
            kill_process(current_process)

