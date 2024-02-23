# Trackers Evaluation on MOT Challenge Data

An easy way to run Similari trackers and measure their performance on real data from 
[Multiple Object Tracking Benchmark (MOTChallenge)](https://motchallenge.net/).

## Download The Data

For the [MOT20](https://motchallenge.net/data/MOT20/) use one of the links:

* [Get files (no img) only](https://motchallenge.net/data/MOT20Labels.zip) (13.9 MB);
* [Get all data](https://motchallenge.net/data/MOT20.zip) (5.0 GB).

```bash
curl -o MOT20Labels.zip https://motchallenge.net/data/MOT20Labels.zip
```

Extract the label files to the `data/` folder:

```bash
unzip MOT20Labels.zip
mkdir data
mv MOT20Labels data/MOT20

```

The `data` folder structure for `MOT20` should be:
```
MOT20/
  test/
    MOT20-04/
    ...
  train/
    MOT20-01/
      det/
      gt/
      ...
    ...
```

## Build The Docker Image

```bash
docker build -t similari_py_mot -f python/motchallenge/Dockerfile .
```

## Run The Processing and Evaluation

Execute code to process default `data/MOT20/train` data using default Similari SORT that uses 
IoU metric with `0.3` threshold: 

```bash
docker run --rm -v $(pwd)/data:/data similari_py_mot
```

Predefined config file for the canonical (python+numpy) SORT:

```shell
docker run --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/python/motchallenge/confs/original-sort-config.toml.yml:/opt/custom_config.yml \
    similari_py_mot /opt/custom_config.yml
```

Predefined config file for the Similari SORT that uses Mahalanobis metric:

```shell
docker run --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/python/motchallenge/confs/similari-maha-sort-config.toml.yml:/opt/custom_config.yml \
    similari_py_mot /opt/custom_config.yml
```


## Interpreting The Results

The whole process is logged on the stdout. 

The FPS processing performance is displayed for every processed file like:

```
Processing MOT20-01...
Read file "/data/MOT20/train/MOT20-01/det/det.txt".
429 frames collected, with an average of 29.4 detections per frame.

100%|██████████| 429/429 [00:00<00:00, 1166.01it/s]MOT20-01 processing ended after 0:00:00.368002 (1165.76 FPS).
Resulting file /data/MOT20/output/sort_iou/data/MOT20-01.txt was successfully written.
```

The resulting files will be saved to the specified folder (`output_path` in the config file).

The `data/MOT20/output/<NAME>/data` folder output structure:
* `pedestrian_detailed.csv` - detailed evaluation info;
* `pedestrian_summary.txt` - evaluation summary;
* `processing_stats.csv` - processing statistics: frames, detections, FPS, etc.
