import socket
import struct
import numpy as np
import datetime
from datetime import datetime as dt
import pickle

N_SECONDS = 65
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

class UDPListener:

    def __init__(self, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096, fileroot='test'):

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)
        self.fileroot = fileroot

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

    def write_to_file(self, all_data, packet_num_all, byte_count_all, num_chunks):
        to_store = (all_data, packet_num_all, byte_count_all)

        d = int(dt.timestamp(dt.utcnow())*1e6)
        with open(self.fileroot + '_' + str(d) +'.pkl', 'wb') as f:
            pickle.dump(to_store, f)

    def read(self, timeout=1):
        """ Read in a single packet via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Frame buffer
        all_data = []
        packet_num_all = []
        byte_count_all = []

        packet_in_chunk = 0
        num_all_packets = 0
        num_chunks = 0

        s_time = dt.utcnow()
        start_time = s_time.isoformat()+'Z'


        try:
            while True:
                packet_num, byte_count, packet_data = self._read_data_packet()
                all_data.append(packet_data)
                packet_num_all.append(packet_num)
                byte_count_all.append(byte_count)
                packet_in_chunk += 1
                num_all_packets += 1

                #### Writing to disk
                # if packet_in_chunk > 2000000:
                #     self.write_to_file(all_data, packet_num_all, byte_count_all, num_chunks)                    

                #     all_data = []
                #     packet_num_all = []
                #     byte_count_all = []
                #     packet_in_chunk = 0
                #     num_chunks += 1     

                #### Stopping after n seconds
                curr_time = dt.utcnow()
                if (curr_time - s_time) > datetime.timedelta(seconds=N_SECONDS):
                    end_time = dt.utcnow().isoformat()+'Z'
                    print("Total packets captured ", num_all_packets)
                    return (all_data, packet_num_all, byte_count_all, start_time, end_time)


        except socket.timeout:
            end_time = dt.utcnow().isoformat()+'Z'
            print("Total packets captured ", num_all_packets)
            # self.write_to_file(all_data, packet_num_all, byte_count_all, num_chunks)

            return (all_data, packet_num_all, byte_count_all, start_time, end_time)
            # return (start_time, end_time)
            pass

        except KeyboardInterrupt:
            end_time = dt.utcnow().isoformat()+'Z'
            print("Total packets captured ", num_all_packets)
            # self.write_to_file(all_data, packet_num_all, byte_count_all, num_chunks)
            return (all_data, packet_num_all, byte_count_all, start_time, end_time)
            # return (start_time, end_time)
            pass

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        return packet_num, byte_count, packet_data

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()