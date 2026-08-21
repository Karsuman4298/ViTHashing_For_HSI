import numpy as np

from data.houston_dataset import stratified_split


def _min_distance_from_query_to_other(valid_positions, query_idx, other_idx):
    q = valid_positions[query_idx]
    others = valid_positions[other_idx]
    if len(others) == 0:
        return np.inf
    dy = others[:, 0] - q[0]
    dx = others[:, 1] - q[1]
    return np.sqrt(dy * dy + dx * dx).min()


def test_spatial_split_has_no_query_overlap_with_train_or_database():
    height, width = 240, 240
    labels = np.zeros((height, width), dtype=np.int64)

    labels[20:80, 20:80] = 1
    labels[110:170, 110:170] = 2
    labels[20:80, 150:210] = 3
    labels[150:210, 30:90] = 4

    split = stratified_split(labels, train_fraction=0.5, query_per_class=20)
    valid_positions = np.argwhere(labels > 0)

    for q_idx in split["query"]:
        q_pos = valid_positions[q_idx]
        min_train = _min_distance_from_query_to_other(valid_positions, q_idx, split["train"])
        min_db = _min_distance_from_query_to_other(valid_positions, q_idx, split["database"])
        assert min_train > 13, (q_pos, min_train)
        assert min_db > 13, (q_pos, min_db)
