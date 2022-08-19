# Tracker evaluation on MOT Challenge data
Simple way to run Similari trackers and measure their performance using real data from 
[Multiple Object Tracking Benchmark (MOTChallenge)](https://motchallenge.net/).

## 1. Download data
E.g. for [MOT20](https://motchallenge.net/data/MOT20/) use one of links
* [Get all data](https://motchallenge.net/data/MOT20.zip) (5.0 GB)
* [Get files (no img) only](https://motchallenge.net/data/MOT20Labels.zip) (13.9 MB)

Unzip data to `data/` folder. 

`data/` folder structure for MOT20 should be
* MOT20/
  * test/
    * MOT20-04/
    * ...
  * train/
    * MOT20-01/
      * det/
      * gt/
      * ...
    * ...

## 2. Build docker image
```shell
docker build -t similari_py_mot -f python/motchallenge/Dockerfile .
```

## 3. Prepare config file
Use [config.yml](config.yml) as a template. E.g. create `custom_config.yml` in the project root folder. 

*Note: Step is optional, presented [config.yml](config.yml) will be used by default.

## 4. Run processing and evaluation
Docker entrypoint is `python3 -m motchallenge /opt/python/motchallenge/config.yml`.
Execute code to process default `data/MOT20/train` data using Sort(IoU). 
```shell
docker run --rm \
    -v `pwd`/data:/data \
    similari_py_mot
```
Or use custom config file  
```shell
docker run --rm \
    -v `pwd`/data:/data \
    -v `pwd`/custom_config.yml:/opt/custom_config.yml \
    similari_py_mot /opt/custom_config.yml
```

## 5. Analyze results
The whole process is logged on the stdout. 

Additionally, the resulting files will be saved to the specified folder (`output_path` in the config file).
Folder structure
* `data\`  *# tracker output in MOTChallenge format*
* `pedestrian_detailed.csv`  *# detailed evaluation info*
* `pedestrian_summary.txt`   *# evaluation summary*
* `processing_stats.csv`     *# processing statistics: frames, detections, FPS..*