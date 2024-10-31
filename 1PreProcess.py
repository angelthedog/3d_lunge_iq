import numpy as np
import csv
import re
import os

def read_bvh_data(file_path, user_index, y1, y2, y3):
    """
    Reads a .bvh file and constructs an array with time series data including user_index, timestamps, 159 joint data points,
    and the three target values (y1, y2, y3).

    Parameters:
    - file_path: str, path to the .bvh file.
    - user_index: int, unique identifier for each user.
    - y1, y2, y3: float, target values for alignment, speed, and precision.

    Returns:
    - data: np.array, shape (N, 164), where each row is [user_index, timestamp, 159 joint data points, y1, y2, y3].
    - frames: int, number of frames in the motion data.
    - frame_time: float, time per frame.
    """
    data = []
    frames = None
    frame_time = None

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # Locate the 'MOTION' keyword
        motion_index = next((i for i, line in enumerate(lines) if line.strip() == 'MOTION'), None)
        if motion_index is None:
            raise ValueError(f"'MOTION' section not found in {file_path}")
        
        # Read the Frames and Frame Time values
        frames_line = lines[motion_index + 1].strip().split()
        if frames_line[0] == "Frames:":
            frames = int(frames_line[1])

        frame_time_line = lines[motion_index + 2].strip().split()
        if frame_time_line[0] == "Frame" and frame_time_line[1] == "Time:":
            frame_time = float(frame_time_line[2])

        # Initialize timestamp
        timestamp = 0.0

        # Process each line after 'MOTION' to directly construct the data array
        for line in lines[motion_index + 3:]:
            data_points = line.strip().split()
            if len(data_points) == 159:
                row = [user_index, timestamp] + [float(x) for x in data_points] + [y1, y2, y3]
                data.append(row)
                timestamp += frame_time  # Increment timestamp by frame_time

    return np.array(data)


def prepare_dataset(data_file):
    """
    Reads data.txt, parses each .bvh file, and constructs a dataset with user_index, timestamp, and y values (alignment, speed, precision),
    and saves the combined dataset to output.csv.

    Parameters:
    - data_file: str, path to data.txt containing file names and labels.

    Returns:
    - metadata: List of tuples (frames, frame_time) for each .bvh file.
    """
    user_index = 1

    with open(data_file, 'r') as f, open("output.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Use the first file in data_file to get the joint names for header construction
        first_bvh_file = f.readline().strip().split(',')[0]
        f.seek(0)
        joints = extract_joints(os.path.join("bvh", first_bvh_file))

        # Construct header with joint names and their axes
        header = ["user_index", "timestamp", "hips_JNT_x", "hips_JNT_y", "hips_JNT_z", "hips_JNT_xR", "hips_JNT_yR", "hips_JNT_zR"]
        for joint in joints:
            header.extend([f"{joint}_xR", f"{joint}_yR", f"{joint}_zR"])
        header += ["alignment", "speed", "precision"]
        writer.writerow(header)

        for line in f:
            parts = line.strip().split(',')
            file_path = os.path.join("bvh", parts[0].strip())
            y_values = list(map(float, parts[1:4]))  # alignment, speed, precision
            
            # Read .bvh file and construct the data array
            data = read_bvh_data(file_path, user_index, *y_values)

            # Write data to CSV
            writer.writerows(data)

            # Increment user index for the next file
            user_index += 1


def extract_joints(file_path):
    """
    Extracts joint names from a .bvh file that start with 'JOINT' and end with '_JNT'.
    
    Parameters:
    - file_path: str, path to the .bvh file.
    
    Returns:
    - joints: List of joint names as strings.
    """
    joints = []

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'JOINT (\S+_JNT)', line)
            if match:
                joint_name = match.group(1)
                joints.append(joint_name)

    return joints


prepare_dataset("input.txt")

