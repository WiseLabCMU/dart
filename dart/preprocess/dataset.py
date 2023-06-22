"""Radar Data Marshalling."""

import numpy as np

from beartype.typing import NamedTuple
from jaxtyping import Complex64, Int16, Float64, Float16, Float

from .radar import AWR1843Boost
from .trajectory import Trajectory
from dart import types


class AWR1843BoostDataset(NamedTuple):
    """Radar raw dataset."""

    start: int
    end: int
    start_packet: int
    frame_size: int
    packet_size: int
    num_packets: int

    @classmethod
    def from_packets(cls, packets, frame_size: int):
        """Initialize from packet dictionary / h5file.

        Parameters
        ----------
        packets: hdf5 dataset/group.
        frame_size: Number of entries per frame; each entry is 4 bytes
            (int16 real and imaginary parts of the IQ stream).
        """
        packet_num = packets['packet_num']
        num_packets = np.max(packet_num) - np.min(packet_num) + 1

        byte_count = packets["byte_count"]

        # 4 bytes per frame entry.
        frame_bytes = frame_size * 4
        first_frame = np.ceil(byte_count[0] / frame_bytes).astype(int)
        last_frame = np.floor(byte_count[-1] / frame_bytes).astype(int)
        # Start, end indices are measured in int16s
        start = int(first_frame * frame_size * 2 - byte_count[0] / 2)
        end = int(last_frame * frame_size * 2 - byte_count[0] / 2)

        return cls(
            start=start, end=end, start_packet=np.min(packet_num),
            packet_size=packets["packet_data"].shape[1],
            frame_size=frame_size, num_packets=num_packets)

    def _get_valid(self, packets):
        """Get valid frame mask."""
        valid = np.zeros(self.num_packets, dtype=bool)
        valid[packets["packet_num"] - self.start_packet] = True
        valid = np.repeat(valid, self.packet_size)
        valid_frames = np.all(
            valid[self.start:self.end].reshape(-1, self.frame_size), axis=1)
        return valid_frames

    def _get_frames(self, packets, valid):
        """Get frames."""
        rad = np.zeros((self.num_packets, self.packet_size), dtype=np.int16)
        rad[packets["packet_num"] - self.start_packet] = packets['packet_data']
        rad = rad.reshape(-1)
        res = rad[self.start:self.end].reshape(-1, self.frame_size)[valid]
        return res

    def _get_times(self, packets, valid) -> Float64[np.ndarray, "frames"]:
        """Get timestamps for each frame.

        Timestamps are denoted by the first packet corresponding to data
        from this frame, where the first packet is assumed to be closest
        to the actual time where the radar chirped.
        """
        start = np.arange(valid.shape[0]) * self.frame_size + self.start
        frame_start_packet = np.floor(start / self.packet_size).astype(int)

        timestamps = np.zeros(self.num_packets, dtype=np.float64)
        timestamps[packets["packet_num"] - self.start_packet] = packets['t']
        return timestamps[frame_start_packet][valid]

    def _to_iq(
        self, radar: AWR1843Boost, frames: Int16[np.ndarray, "frames len"]
    ) -> Complex64[np.ndarray, "frames tx rx chirp"]:
        """Convert frames to a complex IQ array."""
        iq = np.zeros((frames.shape[0], radar.frame_size), dtype=np.complex64)
        iq[:, 0::2] = frames[:, 0::4] + 1j * frames[:, 2::4]
        iq[:, 1::2] = frames[:, 1::4] + 1j * frames[:, 3::4]
        return iq.reshape((-1, *radar.frame_shape))

    def as_chirps(
        self, radar: AWR1843Boost, packets
    ) -> tuple[
        Complex64[np.ndarray, "frames tx rx chirp"],
        Float64[np.ndarray, "frames"]
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
        data = self._get_frames(packets, valid)
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

        return rda, pose
