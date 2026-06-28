import numpy as np

from soe_vinorm.phobert_crf import _viterbi_decode_numpy


def test_viterbi_decode_numpy_uses_masked_lengths():
    emissions = np.array(
        [
            [
                [0.1, 2.0],
                [3.0, 0.2],
                [0.0, 5.0],
            ],
            [
                [2.0, 0.1],
                [0.0, 4.0],
                [5.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    mask = np.array(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    start_transitions = np.zeros(2, dtype=np.float32)
    end_transitions = np.zeros(2, dtype=np.float32)
    transitions = np.zeros((2, 2), dtype=np.float32)

    paths = _viterbi_decode_numpy(
        emissions,
        mask,
        start_transitions,
        end_transitions,
        transitions,
    )

    assert paths == [[1, 0, 1], [0, 1]]
