import timeit

from similari import IoUSort, BoundingBox

if __name__ == '__main__':
    sort = IoUSort(shards = 4, bbox_history = 10, max_idle_epochs = 5, threshold = 0.3)
    box = BoundingBox(10., 5., 7., 7.).as_xyaah()
    tracks = sort.predict([box])
    for t in tracks:
        print(t.id)
    sort.skip_epochs(10)
    wasted = sort.wasted()
    #print(wasted[0])
