#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import listener
import organizer_copy as org
import os
import pickle
import sys
import tables as tb


def collect():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--output', '-o',
		help='Output directory (eg. C:/Users/Administrator/Desktop/dartdata/dataset0',
		default='./'
	)
	args = parser.parse_args()

	if not os.path.exists(args.output):
		os.makedirs(args.output)
	outfile = os.path.join(args.output, 'radarpackets.pkl')

	obj = listener.UDPListener()

	input("Press Enter to continue...")

	all_data = obj.read()

	print("Start time: ", all_data[3])
	print("End time: ", all_data[4])

	with open(outfile, 'wb') as f:
		pickle.dump(all_data, f)

	print("Storing collected files in ", outfile)


if __name__ == '__main__':
	collect()
