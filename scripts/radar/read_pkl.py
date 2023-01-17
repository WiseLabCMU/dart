import organizer_copy as org
import pickle
from scipy.io import savemat
import sys

file_name = sys.argv[1]
file_root = file_name[:-4]

f = open('./' + file_name,'rb')
print(f)
s = pickle.load(f)

# o = org.Organizer(s, 64, 4, 3, 512)
o = org.Organizer(s, 1, 4, 3, 512)
frames = o.organize()

print(frames.shape)

import matplotlib.pyplot as plt
import numpy as np

# # w = np.hanning(512)
# ff = np.fft.fft(frames[0][0][0])

# plt.plot(10*np.log10(np.abs(ff)))
# plt.show()

# plt.plot(np.real(frames[0][0][0]))
# plt.show()

savemat(file_root+'.mat',{'frames':frames, 'start_time':s[3], 'end_time':s[4]})

to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}

with open('./' + file_root + '_read.pkl', 'wb') as f:
    pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
