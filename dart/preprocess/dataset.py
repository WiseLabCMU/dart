"""Radar Data Marshalling."""

import numpy as np

from beartype.typing import NamedTuple
from jaxtyping import Complex64, Int16, Float64

from .radar import AWR1843Boost


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
        """Initialize from packet dictionary / h5file."""
        packet_num = packets['packet_num']
        num_packets = np.max(packet_num) - np.min(packet_num) + 1

        byte_count = packets["byte_count"]
        first_frame = np.ceil(byte_count[0] / 2 / frame_size).astype(int)
        last_frame = np.floor(byte_count[-1] / 2 / frame_size).astype(int)
        start = int(first_frame * frame_size - byte_count[0] / 2)
        end = int(last_frame * frame_size - byte_count[0] / 2)

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

    def as_frames(
        self, radar: AWR1843Boost, packets
    ) -> tuple[
        Complex64[np.ndarray, "frames tx rx chirp"],
        Float64[np.ndarray, "frames"]
    ]:
        """Convert packets to frames."""
        valid = self._get_valid(packets)
        data = self._get_frames(packets, valid)
        times = self._get_times(packets, valid)
        return self._to_iq(radar, data), times
