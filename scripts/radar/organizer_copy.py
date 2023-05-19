import numpy as np 

MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

np.set_printoptions(threshold=np.inf,linewidth=325)

class Organizer:

	def __init__(self, all_data, timestamps, num_chirp_loops, num_rx, num_tx, num_samples):
		self.data = np.array(all_data[0]).reshape(-1)
		self.packet_num = np.array(all_data[1])
		self.byte_count = np.array(all_data[2])
		self.timestamps = np.array(timestamps)

		self.num_packets = len(self.byte_count)
		self.num_chirps = num_chirp_loops*num_tx
		self.num_rx = num_rx
		self.num_samples = num_samples

		self.BYTES_IN_FRAME = self.num_chirps * self.num_rx * self.num_samples * 2 * 2
		self.BYTES_IN_FRAME_CLIPPED = (self.BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
		self.UINT16_IN_FRAME = self.BYTES_IN_FRAME // 2
		self.NUM_PACKETS_PER_FRAME = self.BYTES_IN_FRAME // BYTES_IN_PACKET
		print(f'{len(np.unique(timestamps))} unique timestamps out of {len(timestamps)}')

	def iq(self, raw_frame):
		"""Reorganizes raw ADC data into a full frame

		Args:
			raw_frame (ndarray): Data to format
			num_chirps: Number of chirps included in the frame
			num_rx: Number of receivers used in the frame
			num_samples: Number of ADC samples included in each chirp

		Returns:
			ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

		"""
		ret = np.zeros(len(raw_frame) // 2, dtype=np.csingle)

		# Separate IQ data
		ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
		ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
		return ret.reshape((self.num_chirps, self.num_rx, self.num_samples))

	def get_frames(self, start_chunk, end_chunk):

		# if first packet received is not the first byte transmitted
		if self.byte_count[start_chunk] == 0:
			bytes_left_in_curr_frame = 0
			start = start_chunk*(BYTES_IN_PACKET // 2)
		else:
			frames_so_far = self.byte_count[start_chunk] // self.BYTES_IN_FRAME
			bytes_left_in_curr_frame = (frames_so_far+1)*self.BYTES_IN_FRAME - self.byte_count[start_chunk]
			start = (bytes_left_in_curr_frame // 2) + start_chunk*(BYTES_IN_PACKET // 2)

		# find num of frames
		total_bytes = self.byte_count[end_chunk] - (self.byte_count[start_chunk] + bytes_left_in_curr_frame)
		num_frames = total_bytes // (self.BYTES_IN_FRAME)

		ret_frames = np.zeros((num_frames, self.num_chirps, self.num_rx, self.num_samples), dtype=complex)
		ret_frametimes = np.zeros((num_frames, 1))

		# organizing into frames
		for i in range(num_frames):
			frame_start_idx = start + i*self.UINT16_IN_FRAME
			frame_end_idx = start + (i+1)*self.UINT16_IN_FRAME
			frame = self.data[frame_start_idx:frame_end_idx].astype(np.int16)
			ret_frames[i] = self.iq(frame)
			t_idx = start_chunk + (i * self.BYTES_IN_FRAME) // BYTES_IN_PACKET
			ret_frametimes[i] = np.mean(self.timestamps[t_idx : t_idx + self.NUM_PACKETS_PER_FRAME])

		return ret_frames, ret_frametimes


	def organize(self):

		packets_ooo = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) != 1)[0]
		is_not_monotonic = np.where(np.array(self.packet_num[1:])-np.array(self.packet_num[0:-1]) < 0)[0]

		print('Non monotonic packets: ', is_not_monotonic)

		if len(packets_ooo) == 0:
			print('packets in order')
			start_chunk = 0
			ret_frames, ret_frametimes = self.get_frames(start_chunk, -1)

		elif len(packets_ooo) == 1:
			print('1 packet not in order')
			start_chunk = packets_ooo[0]+1
			ret_frames, ret_frametimes = self.get_frames(start_chunk, -1)
			# start_chunk = 0

		else:
			print('Packet num not in order')
			packets_ooo = np.insert(packets_ooo, 0, 1)
			packets_ooo = np.append(packets_ooo, len(self.packet_num)-1)

			print('Packets ooo', packets_ooo)

			print('Number of packets per frame ', self.NUM_PACKETS_PER_FRAME)

			where_44 = 0
			print(where_44)
			diff = []
			for i in range(where_44, len(packets_ooo)-1):
				diff.append(self.packet_num[packets_ooo[i+1]]-self.packet_num[packets_ooo[i]+1])
			
			print('Packets received before atleast 1 loss ', diff)
			print('Total packets received ', np.sum(np.array(diff)))

			diff = []
			for i in range(where_44+1, len(packets_ooo)-1):
				diff.append(self.packet_num[packets_ooo[i]+1]-self.packet_num[packets_ooo[i]])
			
			print('Packets lost before atleast 1 reception ', diff)
			packets_lost = np.sum(np.array(diff))

			packets_expected = self.packet_num[-1]-self.packet_num[packets_ooo[where_44]+1]+1
			print('Total packets lost ', packets_lost)
			print('Total packets expected ', packets_expected)
			print('Fraction lost ', packets_lost/packets_expected)

			new_packets_ooo = []
			start_new_packets_ooo = []
			end_new_packets_ooo = []
			for i in range(where_44, len(packets_ooo)):
				if (packets_ooo[i] - packets_ooo[i-1]) > self.NUM_PACKETS_PER_FRAME*2:
					new_packets_ooo.append(packets_ooo[i-1])
					start_new_packets_ooo.append(packets_ooo[i-1])
					end_new_packets_ooo.append(packets_ooo[i])

			new_packets_ooo = np.append(new_packets_ooo, -1)

			for i in range(len(start_new_packets_ooo)):

				start_chunk = start_new_packets_ooo[i]+1
				end_chunk = end_new_packets_ooo[i]
				curr_frames, curr_frametimes = self.get_frames(start_chunk, end_chunk)

				if i == 0:
					ret_frames = curr_frames
					ret_frametimes = curr_frametimes
				else:
					ret_frames = np.concatenate((ret_frames, curr_frames), axis=0)
					ret_frametimes = np.concatenate((ret_frametimes, curr_frametimes), axis=0)

		return ret_frames, ret_frametimes
