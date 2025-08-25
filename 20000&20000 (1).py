import os
import random
import numpy as np
from scipy.io.wavfile import read
import pywt

np.random.seed(3)

indices = np.arange(0, 40000)
np.random.shuffle(indices)

split = np.split(indices, 2)
ind_has = split[0]
ind_no = split[1]

#files_has = str(os.getcwd())+"\\f1chunks"
#files_no = str(os.getcwd())+"\\f2chunks"
files_has = r'C:\Users\George\PycharmProjects\pythonProject\f1chunks'
files_no = r'C:\Users\George\PycharmProjects\pythonProject\f2chunks'


#neutrino pulse parameters

#increasing a increasing amplitude
a = 1.18825759e4

#increasing b => increases width and amplitude
b = 7.77861166e9

coefs_has = []
times_has = []
filenames_has = []

files = 20000

for s in range(0,files):
        #print(str(os.getcwd() + '\chunk' + str(s) + '.wav'))
        Fs, noise = read(str(os.getcwd())+'\\f1chunks\chunk' + str(s) + '.wav')
        print(str(os.getcwd())+'\\f1chunks\chunk' + str(s) + '.wav')

        #take a random moment and add neutrino
        random.seed() #change seed
        #offset gia na mh kopei o palmos
        n_random = random.randint(int( Fs / 5000), len(noise) - int(Fs / 5000))
        #metatropi simeiou se xronikh stigmh
        neutrino_time = n_random / Fs

        #creating the neutrino pulse
        #Total time vextor -gia akriveia--->cwt_neutrino_noise
        t = np.linspace(0, len(noise) / Fs, len(noise))
        neutrino_pulse = -a * (t - neutrino_time) * np.exp(-b * (t - neutrino_time)**2)

        #apothikeusi xronwn pou mpainei o palmos
        times_has.append(neutrino_time)
        filenames_has.append(str(os.getcwd())+'\\f1chunks\chunk' + str(s) + '.wav')

        #adding the neutrino to the noise
        offset = int(Fs / 5000)

        k = np.random.uniform(low = 0.01, high=0.1)
        C = 0.1 * max(noise[n_random - offset:n_random + offset]) / max(neutrino_pulse)

        noise = noise + C * neutrino_pulse

        #wavelets transform
        #list of frequencies
        dt = 1/Fs
Wavelet = 'gaus1' #the mother wavlet
Scales = np.arange(1, 10)
#the frequencies used
frequencies = pywt.scale2frequency(Wavelet, Scales) / dt

coef, freqs = pywt.cwt(noise, Scales, Wavelet, sampling_period=dt)
coefs_has.append(coef)

filenames_no = []
coefs_no = []
for s in range(0,files):
        print(str(os.getcwd() + '\\f2chunks\chunkf2_' + str(s) + '.wav'))
        Fs, noise = read(str(os.getcwd())+'\\f2chunks\chunkf2_' + str(s) + '.wav')

        filenames_no.append(str(os.getcwd())+'\\f2chunks\chunkf2_' + str(s) + '.wav')

#wavelets transform
#list of frequencies
dt = 1/Fs
Wavelet = 'gaus1' #the mother wavlet
Scales = np.arange(1, 10)
#the frequencies used
frequencies = pywt.scale2frequency(Wavelet, Scales) / dt

coef, freqs = pywt.cwt(noise, Scales, Wavelet, sampling_period=dt)
coefs_no.append(coef)

coefs_has = np.array(coefs_has).transpose((0, 2, 1))
coefs_no = np.array(coefs_no).transpose((0, 2, 1))

np.save('coefs_has.npy', coefs_has)
np.save('coefs_no.npy', coefs_no)

np.save('times_has.npy', np.array(times_has))

import pickle

with open('files_has.p', 'wb') as f:
    pickle.dump(filenames_has, f)

with open('files_no.p', 'wb') as f:
    pickle.dump(filenames_no, f)






































