from pathlib import Path
from typing import Dict, List, Tuple, Union
import csv


def read_detections(
    file_path: Union[str, Path]
) -> Dict[int, List[Tuple[float, float, float, float, float]]]:
    """Reads detections in `motchallenge` (csv) format from specified file.
    :param file_path: File with detections
    :return: Grouped by frame detection bboxes
    """
    frame_detections = {}
    with open(file_path, 'r') as file_obj:
        # row format: frame, id, left, top, width, height, conf, x, y, z
        # all frame numbers, target IDs and bounding boxes are 1-based
        for row in csv.reader(file_obj):
            frame_num = int(row[0])
            if frame_num not in frame_detections:
                frame_detections[frame_num] = []
            frame_detections[frame_num].append(tuple(map(float, row[2:7])))
    return frame_detections


def write_csv(file_path: Union[str, Path], rows: List[Tuple]):
    """Writes csv file."""
    with open(file_path, mode="w", newline="") as res_file:
        csv_writer = csv.writer(res_file, lineterminator="\n")
        csv_writer.writerows(rows)
