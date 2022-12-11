


from beartype.typing import NamedTuple

from dart import pose


class TrainingColumn(NamedTuple):
    """Single column
    
    1156

    Attributes
    ----------
    pose      96
    valid     256 / 8
    weight    4
    doppler   4
    data      256 * 4
    """
    # pose
    # valid
    # weight
    # doppler
    # data
    pass


def make_column(col, d, pose, sensor):
    pass





