from augment import (
    create_elastic_transformation,
    create_identity_transformation,
    create_rotation_transformation,
)

import numpy as np


def test_basics():
    rot_transform = create_rotation_transformation((2, 2), 90)
    id_transform = create_identity_transformation((2, 2))
    el_transform = create_elastic_transformation((2, 2), (1, 1), 5)

    expected_rot_transform = np.array(
        [
            [[0.27703846, 1.171035], [-1.171035, -0.27703846]],
            [[1.171035, -0.27703846], [0.27703846, -1.171035]],
        ],
        dtype=np.float32,
    )
    expected_id_transform = np.array(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]], dtype=np.float32
    )

    assert all(np.isclose(rot_transform, expected_rot_transform).flatten())
    assert all(np.isclose(id_transform, expected_id_transform).flatten())
