from similari import sutherland_hodgman_clip, intersection_area, BoundingBox

if __name__ == '__main__':
    bbox1 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
    bbox2 = BoundingBox(0.0, 0.0, 10.0, 5.0).as_xyaah()

    clip = sutherland_hodgman_clip(bbox1, bbox2)
    print(clip)

    area = intersection_area(bbox1, bbox2)
    print("Intersection area:", area)


    bbox1 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
    bbox2 = BoundingBox(0.0, 0.0, 5.0, 10.0).as_xyaah()
    bbox2.rotate(0.5)

    clip = sutherland_hodgman_clip(bbox1, bbox2)
    print(clip)

    area = intersection_area(bbox1, bbox2)
    print("Intersection area:", area)

