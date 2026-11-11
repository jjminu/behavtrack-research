import cv2
import numpy as np
import pytest
from behavtrack.dlc_bridge import load_training_image,detection_only_annotations


def test_training_images_support_korean_paths_and_correct_color_order(tmp_path):
    folder=tmp_path/'마우스 학습 자료';folder.mkdir();path=folder/'정답 영상.png'
    image=np.zeros((24,32,3),np.uint8);image[:]=(10,50,200)
    cv2.imencode('.png',image)[1].tofile(str(path))
    assert np.array_equal(load_training_image(path,'BGR'),image)
    assert np.array_equal(load_training_image(path,'RGB'),image[:,:,::-1])
    with pytest.raises(ValueError):load_training_image(path,'unsupported')


def test_training_corrupt_image_has_actionable_failure(tmp_path):
    path=tmp_path/'bad.png';path.write_bytes(b'broken image')
    with pytest.raises(ValueError,match='학습 이미지'):load_training_image(path)


def test_box_only_targets_survive_without_fabricating_visible_keypoints():
    anns=[dict(bbox=[10+i*30,20,20,25],category_id=1) for i in range(5)]
    points,unique,boxes,merged=detection_only_annotations(anns,27)
    assert boxes.shape==(5,4) and np.count_nonzero(boxes[:,2])==5
    assert points.shape==(5,27,3) and np.count_nonzero(points)==0
    assert np.all(merged['category_id']==1) and np.all(merged['area']==500)
    with pytest.raises(ValueError):detection_only_annotations([dict(bbox=[1,2,0,3],category_id=1)],27)
