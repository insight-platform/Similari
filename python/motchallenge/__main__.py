"""Entrypoint.
>>> python -m motchallenge [config_file_path]
"""
from datetime import timedelta
from pathlib import Path
from time import time
import sys

from .config import load_config, OriginalSortParams
from .trackers import OriginalSort, SimilariTracker
from .evaluator import evaluate
from .utils import read_detections, write_result


def main(config_file_path: str):
    """Entrypoint.
    :param config_file_path: Configuration file path
    TODO: Replace `print` with `logger.info`
    """
    config = load_config(config_file_path)

    tracker_name = config.name

    data_path = Path(config.data_path)
    output_path = Path(config.output_path)
    res_path = output_path / tracker_name / 'data'
    res_path.mkdir(parents=True, exist_ok=True)

    for folder in data_path.iterdir():
        if not folder.is_dir():
            continue
        sample = folder.stem
        print(f'Processing {sample}...')
        det_file_path = folder / 'det' / 'det.txt'
        res_file_path = res_path / f'{sample}.txt'

        print(f'Read file "{det_file_path}".')
        frame_detections = read_detections(det_file_path)
        num_frames = len(frame_detections)
        avg_dets = sum(map(len, frame_detections.values())) / num_frames
        print(
            f'{num_frames} frames collected, '
            f'with an average of {avg_dets:.1f} detections per frame.'
        )

        # init tracker
        if isinstance(config.tracker, OriginalSortParams):
            tracker = OriginalSort(config.tracker)
        else:
            tracker = SimilariTracker(config.tracker)

        # track and collect result rows
        print('Tracking...')
        start_time = time()

        result_rows = []
        for frame_num in range(1, len(frame_detections) + 1):
            detections = frame_detections[frame_num]
            result_rows.extend(
                [
                    (frame_num,) + row + (-1.0, -1.0, -1.0)
                    for row in tracker.process_frame(frame_num, detections)
                ]
            )

        exec_seconds = time() - start_time
        print(
            f'ended after {timedelta(seconds=exec_seconds)} '
            f'({num_frames / exec_seconds:.2f} FPS).'
        )

        write_result(res_file_path, result_rows)
        print(f"Resulting file {res_file_path} was successfully written.\n")

    evaluate(tracker_name, data_path, output_path)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'motchallenge/config.yml')
