import pytest

from courtvision.data import Box, mot_to_yolo, parse_mot_line


def test_parse_mot_line() -> None:
    box = parse_mot_line("12,7,100,50,40,80,1,1,0.9")
    assert box.frame == 12
    assert box.track_id == 7
    assert box.w == 40
    assert box.visibility == pytest.approx(0.9)

def test_mot_to_yolo() -> None:
    box = Box(frame=1, track_id=2, x=100, y=50, w=40, h=80)
    cx, cy, w, h = mot_to_yolo(box, image_width=200, image_height=200)
    assert cx == pytest.approx(0.6)
    assert cy == pytest.approx(0.45)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.4)

def test_invalid_mot_line() -> None:
    with pytest.raises(ValueError):
        parse_mot_line("1,2,3")
