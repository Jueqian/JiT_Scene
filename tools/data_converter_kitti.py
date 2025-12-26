import numpy as np
import os
import glob
import natsort


TRAIN_SEQUENCES = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
VAL_SEQUENCES = ["08"]
TEST_SEQUENCES = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]


def parse_calibration(filename):
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def load_poses(calib_fname, poses_fname):
    if os.path.exists(calib_fname):
        calibration = parse_calibration(calib_fname)
        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

    poses_file = open(poses_fname)
    poses = []

    for line in poses_file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        if os.path.exists(calib_fname):
            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
        else:
            poses.append(pose)

    return poses

'''
├── data
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 22
'''
class KittiDataConverter:
    def __init__(self, 
                 root_path="./data/semantickitti/", 
                 split = ["train", "val"],
                 train_sequences=TRAIN_SEQUENCES,
                 val_sequences=VAL_SEQUENCES, 
                 test_sequences=TEST_SEQUENCES,
        ):
        self.root_path = root_path
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.split = split

        self.lidar_path_pattern = os.path.join(root_path, "sequences", "xxx", "velodyne")
        self.image_path_pattern = os.path.join(root_path, "sequences", "xxx", "image_2")
        self.label_path_pattern = os.path.join(root_path, "sequences", "xxx", "labels")

        self.map_clean_path = os.path.join(root_path, "sequences", "xxx", "map_clean.pcd")
        self.poses_file = os.path.join(root_path, "sequences", "xxx", "poses.txt")
        self.calib_file = os.path.join(root_path, "sequences", "xxx", "calib.txt")

        self.training_data = []
        self.validation_data = []
        self.test_data = []

        self.load_data_all()


    def load_data_all(self):
        if "train" in self.split:
            print("Loading training data...")
            for sequence in self.train_sequences:
                seq_data = self.load_sequence(sequence)
                self.training_data.extend(seq_data)

        if "val" in self.split:
            for sequence in self.val_sequences:
                seq_data = self.load_sequence(sequence)
                self.validation_data.extend(seq_data)

        if "test" in self.split:
            for sequence in self.test_sequences:
                seq_data = self.load_sequence(sequence)
                self.test_data.extend(seq_data)


    def load_sequence(self, sequence):

        sequence_data = []
        lidar_path = self.lidar_path_pattern.replace("xxx", sequence)
        image_path = self.image_path_pattern.replace("xxx", sequence)
        label_path = self.label_path_pattern.replace("xxx", sequence)
        poses_file = self.poses_file.replace("xxx", sequence)
        calib_file = self.calib_file.replace("xxx", sequence)

        lidar_files = natsort.natsorted(glob.glob(os.path.join(lidar_path, "*.bin")))
        image_files = natsort.natsorted(glob.glob(os.path.join(image_path, "*.png")))
        label_files = natsort.natsorted(glob.glob(os.path.join(label_path, "*.label")))

        map_clean_path = self.map_clean_path.replace("xxx", sequence)

        poses = load_poses(calib_file, poses_file)
        # import ipdb; ipdb.set_trace()

        assert len(lidar_files) == len(image_files) == len(label_files) == len(poses), \
            f"Data length mismatch in sequence {sequence}"
        
        for idx in range(len(lidar_files)):
            frame_info_dict = {
                "frame_id": f"{sequence}_{idx:06d}",
                "lidar_path": lidar_files[idx],
                "image_path": image_files[idx],
                "label_path": label_files[idx],
                "map_clean_path": map_clean_path,

                "pose": poses[idx],
            }
            sequence_data.append(frame_info_dict)
        return sequence_data


    def save_all_data(self, save_dir=None):
        save_dir = self.root_path if save_dir is None else save_dir
        os.makedirs(save_dir, exist_ok=True)
        if "train" in self.split:
            self.save_pkl(os.path.join(save_dir, "semantickitti_train_info.pkl"), self.training_data)
        if "val" in self.split:
            self.save_pkl(os.path.join(save_dir, "semantickitti_val_info.pkl"), self.validation_data)
        if "test" in self.split:
            self.save_pkl(os.path.join(save_dir, "semantickitti_test_info.pkl"), self.test_data)


    def save_pkl(self, save_path, data):
        import pickle
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {save_path}")




if __name__ == "__main__":
    root_path = "/Users/xiaodong/repo/SemanticKITTI"
    converter = KittiDataConverter(root_path, train_sequences=["06", "07"], val_sequences=["06"])
    converter.save_all_data()

