"""Radar Data Marshalling."""

import numpy as np

from beartype.typing import NamedTuple
from jaxtyping import Complex64, Int16, Float64, Float16, Float

from .radar import AWR1843Boost
from .trajectory import Trajectory
from dart import types


class AWR1843BoostDataset(NamedTuple):
    """Radar raw dataset.

    Attributes
    ----------
    start: index of the first chirp.
    end: index of the last chirp.
    """

    start: int
    end: int
    start_packet: int
    chirp_size: int
    packet_size: int
    num_packets: int

    @classmethod
    def from_packets(cls, packets, chirp_size: int):
        """Initialize from packet dictionary / h5file.

        Parameters
        ----------
        packets: hdf5 dataset/group.
        chirp_size: Number of entries per chirp; each entry is 4 bytes
            (int16 real and imaginary parts of the IQ stream).
        """
        packet_num = packets['packet_num']
        num_packets = np.max(packet_num) - np.min(packet_num) + 1

        byte_count = packets["byte_count"]

        # 4 bytes (int16 IQ) per chirp entry.
        chirp_bytes = chirp_size * 4
        first_chirp = np.ceil(byte_count[0] / chirp_bytes).astype(int)
        last_chirp = np.floor(byte_count[-1] / chirp_bytes).astype(int)
        # Start, end indices are measured in int16s
        start = int(first_chirp * chirp_size * 2 - byte_count[0] / 2)
        end = int(last_chirp * chirp_size * 2 - byte_count[0] / 2)

        return cls(
            start=start, end=end, start_packet=np.min(packet_num),
            packet_size=packets["packet_data"].shape[1],
            chirp_size=chirp_size, num_packets=num_packets)

    def _get_valid(self, packets):
        """Get valid chirp mask.

        Valid chirps are defined as chirps where each packet that would make up
        the chirp has not been dropped.
        """
        valid = np.zeros(self.num_packets, dtype=bool)
        valid[packets["packet_num"] - self.start_packet] = True
        valid = np.repeat(valid, self.packet_size)
        valid_chirps = np.all(
            valid[self.start:self.end].reshape(-1, self.chirp_size * 2),
            axis=1)
        return valid_chirps

    def _get_chirps(self, packets, valid):
        """Get chirp data as a int16 array."""
        rad = np.zeros((self.num_packets, self.packet_size), dtype=np.int16)
        rad[packets["packet_num"] - self.start_packet] = packets['packet_data']
        rad = rad.reshape(-1)
        res = rad[self.start:self.end].reshape(-1, self.chirp_size * 2)[valid]
        return res

    def _get_times(self, packets, valid) -> Float64[np.ndarray, "chirps"]:
        """Get timestamps for each chirp.

        Timestamps are denoted by the first packet corresponding to data
        from this chirp, where the first packet is assumed to be closest
        to the actual time where the radar chirped.
        """
        start = np.arange(valid.shape[0]) * self.chirp_size * 2 + self.start
        start_packet = np.floor(start / self.packet_size).astype(int)

        timestamps = np.zeros(self.num_packets, dtype=np.float64)
        timestamps[packets["packet_num"] - self.start_packet] = packets['t']
        return timestamps[start_packet][valid]

    def _to_iq(
        self, radar: AWR1843Boost, raw: Int16[np.ndarray, "chirps len"]
    ) -> Complex64[np.ndarray, "chirps tx rx chirp"]:
        """Convert chirps to a complex IQ array."""
        iq = np.zeros((raw.shape[0], radar.chirp_size), dtype=np.complex64)
        iq[:, 0::2] = raw[:, 0::4] + 1j * raw[:, 2::4]
        iq[:, 1::2] = raw[:, 1::4] + 1j * raw[:, 3::4]
        return iq.reshape((-1, *radar.chirp_shape))

    def as_chirps(
        self, radar: AWR1843Boost, packets
    ) -> tuple[
        Complex64[np.ndarray, "chirps tx rx chirp"],
        Float64[np.ndarray, "chirps"]
    ]:
        """Convert packets to chirps.

        Parameters
        ----------
        radar: Radar object.
        packets: input data hdf5 dataset/group.

        Returns
        -------
        iq: IQ complex-value chirps for processing.
        times: Timestamps of each chirp.
        """
        valid = self._get_valid(packets)
        data = self._get_chirps(packets, valid)
        times = self._get_times(packets, valid)
        return self._to_iq(radar, data), times

    def process_data(
        self, radar: AWR1843Boost, trajectory: Trajectory, packets
    ) -> tuple[
        Float16[types.ArrayLike, "idx antenna range doppler"],
        dict[str, Float[types.ArrayLike, "idx ..."]]
    ]:
        """Process dataset.

        Parameters
        ----------
        radar: Radar object with radar processing routines.
        trajectory: Dataset trajectory for pose interpolation / smoothing.
        packets: Radar packet dataset.

        Returns
        -------
        rda: Processed range-doppler-azimuth images.
        pose: Pose information (position, rotation, velocity, time) formatted
            as a dictionary.
        """
        chirps, t_chirp = self.as_chirps(radar, packets)
        rda, speed_radar = radar.process_data(chirps)
        t_image = radar.process_timestamps(t_chirp)

        window_size = radar.frame_time * 0.5
        t_valid = trajectory.valid_mask(t_image, window=window_size)
        pose = trajectory.interpolate(t_image[t_valid], window=window_size)
        pose["t"] = t_image[t_valid]
        pose["speed"] = speed_radar[t_valid]

        return rda[t_valid], pose
