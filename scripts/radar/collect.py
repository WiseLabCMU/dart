import listener
# import organizer
import sys
import pickle
import glob

filename = sys.argv[1]
fileroot = filename[:-4]

obj = listener.UDPListener(fileroot=fileroot)

input("Press Enter to continue...")

all_data = obj.read()

print("Start time: ", all_data[3])
print("End time: ", all_data[4])

with open(filename, 'wb') as f:
	pickle.dump(all_data, f)

print("Storing collected files in ", filename)

# print("Start time: ", all_data[0])
# print("End time: ", all_data[1])

# with open(fileroot + '_timestamp.pkl', 'wb') as f:
# 	pickle.dump(all_data, f)

# print("Storing collected files in ", fileroot)