from similari import nms, BoundingBox, Universal2DBox

if __name__ == '__main__':
    bb = BoundingBox(left=1.0, top=2.0, width=10.0, height=15.0)
    print(bb)

    bb = BoundingBox(1.0, 2.0, 10.0, 15.0)
    print(bb.left, bb.top, bb.width, bb.height)

    bb = BoundingBox.new_with_confidence(1.0, 2.0, 10.0, 15.0, 0.95)

    universal_bb = bb.as_xyaah()
    print(universal_bb)

    ubb = Universal2DBox(xc=3.0, yc=4.0, angle=0.0, aspect=1.5, height=5.0)
    print(ubb)

    ubb = Universal2DBox.new_with_confidence(xc=3.0, yc=4.0, angle=0.0, aspect=1.5, height=5.0, confidence=0.85)
    print(ubb)

    ubb = Universal2DBox(3.0, 4.0, 0.0, 1.5, 5.0)
    print(ubb)
    ubb.rotate(0.5)

    polygon = ubb.get_vertices()
    points = polygon.get_points()
    print("Points", points)

    print(ubb)
    print(ubb.area())
    print(ubb.get_radius())
