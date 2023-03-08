import pathlib
import collections
import json
from similari import (
    VisualSort,
    SpatioTemporalConstraints,
    PositionalMetricType,
    VisualSortOptions,
    VisualSortMetricType,
    Universal2DBox, VisualSortObservation, VisualSortObservationSet
)


def main():
    data_dir_path = pathlib.Path('.') / 'in'
    out_dir_path = pathlib.Path('.') / 'out'
    log_file_path = out_dir_path / 'log.txt'

    constraints = SpatioTemporalConstraints()
    constraints.add_constraints([(1, 1.0)])

    opts = VisualSortOptions()
    opts.spatio_temporal_constraints(constraints)
    opts.max_idle_epochs(3)
    opts.kept_history_length(10)
    # opts.visual_metric(VisualSortMetricType.cosine(0.2))
    opts.visual_metric(VisualSortMetricType.euclidean(1.0))
    opts.positional_metric(PositionalMetricType.maha())
    opts.visual_minimal_track_length(1)
    opts.visual_minimal_area(5.0)
    opts.visual_minimal_quality_use(0.45)
    opts.visual_minimal_quality_collect(0.5)
    opts.visual_max_observations(5)
    opts.visual_min_votes(1)

    tracker = VisualSort(shards=4, opts=opts)

    with open(log_file_path, 'w', encoding='utf8') as log_file:

        for filepath in sorted(data_dir_path.glob('*.json')):

            with open(filepath, 'r', encoding='utf8') as objfile:
                objs = json.load(objfile)

            observation_set = VisualSortObservationSet()

            for obj in objs:
                bbox = Universal2DBox.new_with_confidence(
                    xc=obj['bbox']['xc'],
                    yc=obj['bbox']['yc'],
                    angle=obj['bbox']['angle'],
                    aspect=obj['bbox']['aspect'],
                    height=obj['bbox']['height'],
                    confidence=obj['bbox']['confidence']
                )
                observation = VisualSortObservation(
                    feature=obj['feature'],
                    feature_quality=obj['feature_quality'],
                    bounding_box=bbox,
                    custom_object_id=None,
                )
                observation_set.add(observation)

            tracks = tracker.predict(observation_set)

            log_file.write(f'===={filepath.name}====\n')

            for track in tracks:
                log_file.write(str(track) + '\n')

            count = collections.Counter((track.id for track in tracks))
            trk_id, id_count = count.most_common(1)[0]
            assert id_count == 1, f'{filepath.name} trk id {trk_id} count {id_count}'


if __name__ == '__main__':
    main()