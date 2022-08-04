from similari import nms, parallel_nms, BoundingBox

if __name__ == '__main__':

    print("Sequential - with score")
    bbox1 = (BoundingBox(10.0, 11.0, 3.0, 3.8).as_xyaah(), 1.0)
    bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), 0.9)
    res = nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
    print(res[0].as_xywh())

    print("Sequential - no score")
    bbox1 = (BoundingBox(10.0, 11.0, 3.0, 4.0).as_xyaah(), None)
    bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), None)
    res = nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
    print(res[0].as_xywh())

    print("Parallel - with score")
    bbox1 = (BoundingBox(10.0, 11.0, 3.0, 3.8).as_xyaah(), 1.0)
    bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), 0.9)
    res = parallel_nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
    print(res[0].as_xywh())

    print("Parallel - no score")
    bbox1 = (BoundingBox(10.0, 11.0, 3.0, 4.0).as_xyaah(), None)
    bbox2 = (BoundingBox(10.3, 11.1, 2.9, 3.9).as_xyaah(), None)
    res = parallel_nms([bbox2, bbox1], nms_threshold = 0.7, score_threshold = 0.0)
    print(res[0].as_xywh())
