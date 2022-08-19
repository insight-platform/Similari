from pathlib import Path
import trackeval
from .config import Evaluator


def evaluate(
    tracker_name: str, data_path: Path, output_path: Path, eval_conf: Evaluator
):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['PLOT_CURVES'] = True
    if eval_conf.num_cores > 1:
        eval_config['USE_PARALLEL'] = True
        eval_config['NUM_PARALLEL_CORES'] = eval_conf.num_cores
    eval_config['LOG_ON_ERROR'] = './eval_error_log.txt'
    evaluator = trackeval.Evaluator(eval_config)

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = str(data_path)
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['SEQ_INFO'] = {
        sub.name: None for sub in data_path.iterdir() if sub.is_dir()
    }
    dataset_config['TRACKERS_FOLDER'] = str(output_path)
    dataset_config['TRACKERS_TO_EVAL'] = [tracker_name]
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

    metrics_list = [
        # trackeval.metrics.HOTA(),
        # Similarity score threshold required for a TP match. Default 0.5.
        trackeval.metrics.CLEAR(config={'THRESHOLD': 0.5}),
        # Similarity score threshold required for a IDTP match. Default 0.5.
        trackeval.metrics.Identity(config={'THRESHOLD': 0.5}),
    ]

    evaluator.evaluate(dataset_list, metrics_list)
