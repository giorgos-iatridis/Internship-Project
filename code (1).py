import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import soundfile
from scipy.io.wavfile import read
import pywt
from scipy.fftpack import fft
import array as arr
import glob
import csv

Scales= np.arange(0.1,20)
files=20000
C = np.array([0,150])
z0 = [[-3000] * len(C)*len(Scales) for x in range(files)]
z = [-15000 for x in range(files)]

Wavelet="morl"

#Scales= np.arange(1,10)
#the frequencies used
print("SCALES=", Scales)
print('length of Scales =', len(Scales))

for s in range(0,files):
        #print(str(os.getcwd() + '\chunk' + str(s) + '.wav'))
        Fs, noise_ref = read(str(os.getcwd())+'\\f1chunks\chunk' + str(s) + '.wav')
        #print('fs=',Fs,'noise=', noise_ref)

#take a random moment and add neutrino
        t = np.linspace(0, len(noise_ref) / Fs, len(noise_ref))
        #print(len(t))
        random.seed()  # change seed
        n_random = random.randint(150, len(t)-150)
    #print ("n_random=",n_random)
        neutrino_time = n_random / Fs

#neutrino_time=0.0025
   # print('the neutrino is put in time : %f ' % (neutrino_time))

# Neutrino pulse parameters
        a = 1.18825759e+04
        b = 7.77861166e+09
# creating the neutrino pulse

        neutrino_pulse = -a*( t - neutrino_time )*np.exp(-b*(t-neutrino_time )**2)
        new_pulse =( 1 / (np.pi * (t - neutrino_time)) * (np.sin(20000 * np.pi * (t - neutrino_time)) - np.sin(10000 * np.pi * (t - neutrino_time))))*5*0.000001
        new_pulse[:np.digitize(neutrino_time, t) - 25] = 0
        new_pulse[np.digitize(neutrino_time, t) + 50:] = 0


        offset = int(Fs / 15000)
        #print("file*****",s)
        for m in range(0,len(C)):
            #print("C position", m)
            noise = noise_ref + C[m] * new_pulse
            dt=1./Fs
            frequencies = pywt.scale2frequency(Wavelet, Scales) / dt
            coef, freqs = pywt.cwt(noise, Scales, Wavelet, sampling_period=dt)

            max = [-3000. for x in range(len(freqs))]
            min = [15000. for x in range(len(freqs))]

            for i in range(0, len(freqs)):

                for j in range(100,len(t)-100):


                    if max[i] < coef[i, j]:
                        max[i] = coef[i, j]
                        x = j
                    if min[i] > coef[i, j]:
                        min[i] = coef[i, j]
                        y = j

                z0[s][i + m *len(Scales)] = max[i] - min[i]

                #print('Wavelet',i)
                #print("max in time",x/Fs,"s =",max[i])
                #print("min in time",y/Fs,"s =",min[i])
                #print("***", z0[s][i + m * 9])

for i in range(0,len(Scales)):
       #print("freqs", freqs[i], "*****i=", i)
       plt.figure(i)

       plt.title("Frequency %i Hz Morlet " % freqs[i]  )
       plt.ylabel('files')
       plt.xlabel('max-min')
       for m in range(0, len(C)):

            for s in range(0, files):

                    z[s]=z0[s][i+m*len(Scales)]

            #print("z**", z)
            bins, edges = np.histogram(z,100,range=[0,150])
            #labels = [" %s " % C[m]]
            left, right = edges[:-1], edges[1:]
            X = np.array([left, right]).T.flatten()
            Y = np.array([bins, bins]).T.flatten()
            plt.plot(X, Y,label='C=%s' %C[m])
       plt.legend()
       #print("z**",z)

plt.show()