import os
import glob
from pydub import AudioSegment
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
#import librosa
import librosa.display 
from numpy.lib import stride_tricks
from scipy import signal

import scipy.io.wavfile as wav
import warnings
import wave
import struct
import numpy as np
import math
'''
import os
import glob
import numpy as np

from matplotlib import pyplot as plt

import scipy.io.wavfile as wav
from scipy.fftpack import fft
from scipy import signal

import librosa
import IPython.display as ipd

import matplotlib.pyplot as plt
from librosa.display import specshow, waveplot


#sample = wave.open('103_A_Sojourn_01_380.wav', 'r')'''

'''
import soundfile as sf 
data, fs = sf.read('103_A_Sojourn_01_380.wav') 
counter = 50
for i in data:
	if counter>0:
		#print(i)
		print((math.asin(i[0]/  32765) / 6.28) * 44100 , (math.asin(i[1]/  32765) / 6.28) * 44100 , i[0], i[1])
		counter -= 1
'''


'''
sample = wave.open('103_A_Sojourn_01_380.wav', 'r')
for i in range(10):
	print(int(sample.readframes(i)))
'''

# частота в герцах нот первой октавы - звуки с частотами от [261.63, 523.25)
#                      до      ре      ми     фа       соль    ля      си
freq_array = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88])
# little_octave 
yellow = np.array([65.406, 73.416, 82.407, 87.307, 97.999, 110.00, 123.47])
# big_octave 
#green = np.array([65.406, 73.416, 82.407, 87.307, 97.999, 110.00, 123.47])
#green = np.array([130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94])
green = np.array([261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88])

#	duration
#place	note
composition1 = [[ 0, 8, 0],
[8, 12, green[1]],
[12, 16, (green[1]+green[2])/2],
[16, 24, (green[4]+green[5])/2],
[19, 23, (green[1]+green[2])/2],
[23, 29, green[1]],
[34, 37, (green[1]+green[2])/2],
[37, 41, green[1]],
[41, 44, (green[4]+green[5])/2],
[44, 47, (green[1]+green[2])/2],
[47, 51, green[1]],
[54, 58, green[1]],
[58, 62, green[3]],
[62, 65, (green[4]+green[5])/2],
[65, 69, green[3]],
[69, 73, green[1]],
[73, 77, (green[4]+green[5])/2],
[77, 80, green[3]],
[80, 84, green[1]],
[84, 88, (green[4]+green[5])/2],
[88, 92, green[3]],
[92, 116, (green[4]+green[5])/2],
[104, 107, green[3]],
[120, 130, green[3]],
[120, 130, (green[4]+green[5])/2],
[130, 135, green[4]],
[136, 141, green[4]],
[142, 147, green[4]],
[147, 152, green[4]],
[153, 158, green[4]],
[153, 158, (green[5]+green[6])/2],
[158, 166, (green[5]+green[6])/2],
[158, 166, (green[4]+green[5])/2],
[158, 166, (green[1]+green[2])/2],
[166, 169, green[4]],
[169, 172, green[3]],
[172, 176, (green[1]+green[2])/2],
[178, 181, green[0]],
[181, 185, (green[1]+green[2])/2],
[185, 195, (green[4]+green[5])/2],
[195, 199, (green[4]+green[5])/2],
[188, 192, (green[1]+green[2])/2],
[192, 195, green[0]],
[199, 203, (green[1]+green[2])/2],
[203, 206, green[0]],
[206, 210, (green[4]+green[5])/2],
[210, 213, (green[1]+green[2])/2],
[213, 216, green[0]],
[219, 222, green[0]],
[222, 225, green[3]],
[225, 235, (green[4]+green[5])/2],
[228, 231, green[3]],
[231, 234, green[0]],
[235, 238, green[3]],
[238, 241, green[0]],
[241, 244, (green[3]+green[5])/2],
[244, 247, green[3]],
[247, 250, green[0]],
[250, 253, green[3]],
[253, 256, (green[4]+green[5])/2],
[259, 262, (green[4]+green[5])/2],
[262, 265, green[3]],
[268, 271, (green[4]+green[5])/2],
[271, 274, green[3]],
[274, 281, green[3]],
[274, 281, (green[4]+green[5])/2],
[284, 287, green[4]],
[287, 290, (green[5]+green[6])/2],
[293, 299, (green[5]+green[6])/2],
[298, 300, green[3]],
[300, 308, (green[1]+green[2])/2],
[300, 308, (green[4]+green[5])/2],
[300, 308, (green[5]+green[6])/2]]

# частота дискретизации
SAMPLE_RATE = 44100
# 16-ти битный звук (2 ** 16 -- максимальное значение для int16)
S_16BIT = 2 ** 16

def generate_sample(creation, freq, place, duration, volume):
	#print('generate_sample call', freq, place, duration, volume)
	# амплитуда
    amplitude = np.round(S_16BIT * volume)
    # длительность генерируемого звука в сэмплах
    total_samples = np.round(SAMPLE_RATE * duration)
    # частоте дискретизации (пересчитанная)
    w = 2.0 * np.pi * freq / SAMPLE_RATE
    # массив сэмплов
    k = np.arange(0, total_samples)

    for i in range(place-len(creation) + duration + 1):
    	creation.append(0)

    for x in range(int(duration  * (SAMPLE_RATE / 1000.0))):
    	creation.append(volume * np.sin(2 * np.pi * freq * ( x / SAMPLE_RATE)))
    return creation

def generic_note(creation, freq, duration, volume = 1.0, octave = 1):
	freq *= octave
	duration  *= 2
	for x in range(int(duration  * (SAMPLE_RATE / 1000.0))):
		y = (S_16BIT * volume * np.sin(2 * np.pi * freq * ( x / SAMPLE_RATE))).astype('int16')
		creation.append(y)
	return creation

def generate_tones(creation, attitude, duration, attenuation, max_volume):
	print('generate_tones call', len(attitude), duration, attenuation)
	step = max_volume/ attenuation
	volume = 0

	for i in range(len(attitude)):
		if i < attenuation:
			volume += step
		if i > duration - attenuation:
			volume -= step

		creation = generic_note(creation, attitude[i][2], \
			(attitude[i][1]-attitude[i][0])*50, volume)

	return creation

def write_wave(Name, Frames):
	file = wave.open(Name, 'w')
	file.setparams((1, 2, SAMPLE_RATE, len(Frames), 'NONE', 'not compressed'))
	#file.setparams((1, 2, SAMPLE_RATE, len(Frames)*2, 'NONE', 'not compressed'))
	result = []
	for frame in Frames:
		file.writeframes(struct.pack('h', int(frame)))
	file.close()

def combine(Name, sound1, sound2):
	sound1 = AudioSegment.from_file(path1)
	sound2 = AudioSegment.from_file(path2)
	#sound1 = wave.open(path1, 'r')
	#sound2 = wave.open(path2, 'r')
	#combined = wav.open("combined.wav", w)
	combined = sound1.overlay(sound2)
	combined = combine.fade_in(4000).fade_out(4000)
	combined.export(Name, format='wav')
	sound1.close()
	sound2.close()
	combined.close()
	return combined

def get_grafics(path):
	cnt = []
	for i, audio_path in enumerate(glob.glob(path)):
	    filename = os.path.basename(audio_path)[:-4]
	    x, sr = librosa.load(audio_path)
	    print(x, sr)
	    x = librosa.to_mono(x)
	    
	    cnt.append(len(x) / sr)

	plt.figure(figsize=(20,8))
	plt.hist(x, bins=81)
	plt.title("Duration of sounds", fontsize=20)
	plt.xlabel("Time(s)", fontsize=15)
	plt.ylabel("Number of audio", fontsize=15)
	plt.show()

	plt.figure(figsize=(20,8))
	# plt.plot(samples) 
	librosa.display.waveplot(x, sr=sr)
	plt.title("Audio wave of a \"9\"", fontsize=20)
	plt.xlabel("Frame (8000 frame/seconds)", fontsize=15)
	plt.ylabel("Amplitude", fontsize=15)
	plt.show()  

def get_spectrogramm(path):
	sample_rate, samples = wav.read(path)

	if len(samples.shape) > 1:
	    samples = samples[:,0]

	# scipy
	frequencies, times, spectogram = signal.spectrogram(samples, fs=1)
	#plt.subplot(1, 3, 1)
	plt.imshow(np.log(spectogram), extent=[0,spectogram.shape[1],0,spectogram.shape[0]], aspect='auto')
	plt.ylabel("Freq")
	plt.xlabel("Frame")
	plt.title("Scipy", fontsize=15)
	plt.show()
	        
	# matplotlib
	#plt.subplot(1, 3, 2)
	#plt.subplot(1, 2, 1)
	PxN, freqsN, binsN, imN = plt.specgram(samples, NFFT = 256, Fs=1, Fc=0, window = signal.tukey(256), pad_to = None, noverlap = 1 )
	plt.ylabel("Freq")
	plt.xlabel("Frame")
	plt.title("Matplotlib", fontsize=15)
	plt.show()

	# Librosa
	y, sr = librosa.load(path, sr=None)
	y = librosa.to_mono(y)
	D = librosa.stft(y)  
	#plt.subplot(1, 3, 3)
	#plt.subplot(1, 2, 2)
	# plt.imshow(np.log(np.abs(D)), extent=[0, D.shape[1], 0, D.shape[0]], aspect='auto')
	Xdb = librosa.amplitude_to_db(abs(D))
	librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
	plt.ylabel("Freq")
	plt.xlabel("Frame")
	plt.title("Librosa", fontsize=15)

	plt.show()

# mpl.plot(np.arange(0, len(samples)), samples)
# mpl.grid()
# mpl.show()

flg=2

if flg==0:
	# Audio segment creation 
	creation = [0 for i in range(len(composition1)*2)]
	creation = generate_tones(creation, composition1, len(composition1)*2, 8, 0.3)
	print('generated')
	print(creation)
	write_wave('generated1.wav', creation)
	print('written')

elif flg==1:
	# Audio segments combination
	sound1 = AudioSegment.from_file('generated1.wav')
	sound2 = AudioSegment.from_file('C_BowedSteel_SP_304_01.wav')
	combined = sound1.overlay(sound2)
	combined.export('combined1.wav', format='wav')
else: 
	# Getting grafics and spectrogramms to all files
	#get_grafics('generated1.wav')
	#get_spectrogramm('generated1.wav')
	#get_grafics('combined1.wav')
	#get_spectrogramm('combined1.wav')
	#get_grafics('05-Fall-Field-Cricket.wav')
	#get_spectrogramm('05-Fall-Field-Cricket.wav')
	#get_grafics('C_BowedSteel_SP_304_01.wav')
	get_spectrogramm('C_BowedSteel_SP_304_01.wav')
