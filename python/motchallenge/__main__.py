"""Entrypoint.
>>> python -m motchallenge [config_file_path]
"""
from datetime import timedelta
from pathlib import Path
from time import time
import sys

from tqdm import tqdm

from .config import load_config, OriginalSortParams
from .trackers import OriginalSort, SimilariTracker
from .evaluator import evaluate
from .utils import read_detections, write_csv


def main(config_file_path: str):
    """Entrypoint.
    :param config_file_path: Configuration file path
    TODO: Replace `print` with `logger.info`
    """
    config = load_config(config_file_path)

    tracker_name = config.name

    data_path = Path(config.data_path)
    output_path = Path(config.output_path)
    tracker_path = output_path / tracker_name
    res_path = tracker_path / 'data'
    res_path.mkdir(parents=True, exist_ok=True)

    processing_stat_rows = [('sample', 'num_frames', 'avg_dets', 'exec_seconds', 'fps')]
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
        result_rows = []
        frame_iter = tqdm(range(1, len(frame_detections) + 1))
        start_time = time()
        for frame_num in frame_iter:
            detections = frame_detections[frame_num]
            result_rows.extend(
                [
                    (frame_num,) + row + (-1.0, -1.0, -1.0)
                    for row in tracker.process_frame(frame_num, detections)
                ]
            )

        exec_seconds = time() - start_time
        fps = num_frames / exec_seconds
        print(
            f'{sample} processing ended after {timedelta(seconds=exec_seconds)} '
            f'({fps:.2f} FPS).'
        )

        write_csv(res_file_path, result_rows)
        print(f"Resulting file {res_file_path} was successfully written.\n")

        processing_stat_rows.append((sample, num_frames, avg_dets, exec_seconds, fps))

    write_csv(tracker_path / 'processing_stats.csv', processing_stat_rows)

    evaluate(tracker_name, data_path, output_path, config.evaluator)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'motchallenge/config.toml.yml')
