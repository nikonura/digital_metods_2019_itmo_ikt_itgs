from numpy import random
import sklearn
from scipy.stats import entropy
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def generate_sequence_N(mu, dispersion, n):
	# sigma - стандартное отклонение. Квадрат стандартного отклонения, 
	# sigma ^ 2 , называется дисперсией.
	sigma = np.sqrt(dispersion)
	# нужно мат ожидание 0 дисперсия 1
	# loc, mu - среднее значение, центр распределения
	# scale - стандартное отклонение (распространение или «ширина») распределения.
	#distribution = np.random.normal(loc=5.0,scale=1.0,size=100)
	distribution = np.random.normal(mu, sigma, n)
	print(distribution)

	print(abs(mu - np.mean(distribution)))
	print(abs(sigma - np.std(distribution, ddof=1)) <= 1)
	return distribution

def get_histogram(mu, sigma, distribution):
	count, bins, ignored = plt.hist(distribution, 30, density=True)
	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *\
		np.exp( - (bins - mu)**2 / (2 * sigma**2) ),\
		linewidth=2, color='r')
	plt.show()
	return 1

math_wait = 0
dispersion = 1
n = 100
#N_distribution = generate_sequence_N(math_wait, dispersion, n)
origin_dispersion = [ 1.3158006 , -2.45677821,  0.54352045, -0.14106508,  0.23513193, -0.36292628,
  0.90243361, -0.14536643, -1.46801483,  0.44261825,  0.38843632, -1.82063799,
  1.31543399,  0.69356634,  0.16041174,  0.98493171, -0.13551327,  1.23541863,
  0.95210511, -0.06585809, -0.82611768, -0.67309931, -0.76080306, -1.23371526,
 -1.09988864, -1.49680864,  0.3667013 , -1.25905782,  0.30013024,  0.66157705,
 -0.34055059,  0.15382166, -0.66424259,  1.33198847,  1.71607822,  1.48916607,
  1.0117308 ,  1.38634016,  0.86760575, -0.96628115, -1.06085691, -0.83529262,
 -0.93367317, -0.11108838,  0.16953682, -1.14657612,  0.79002362,  1.15747936,
 -1.83179412, -0.45135497,  0.3635317 , -1.57040541,  2.56240298,  2.03667232,
  1.05787374, -0.7516366 ,  0.18537683,  0.13468368,  0.09211   ,  0.45813666,
 -0.28300264, -0.44608182,  0.0939373 ,  0.68842533, -0.22890885, -1.71786003,
  0.63062916,  0.0607422 ,  0.15275705,  1.37494718,  2.03064677,  0.1326493 ,
  1.12692353, -0.79661524, -0.1688151 , -0.15467217,  0.7131292 , -0.29994256,
 -0.56412924, -0.04145376, -0.43046424, -0.63252451, -1.37327877, -2.0999412 ,
 -0.40279749, -0.61871218,  0.10206605, -0.01062984, -2.23605642, -1.61036094,
 -1.48024384,  0.81023954,  1.0326031 ,  1.39715117,  0.7382808 , -0.28619269,
 -0.10121803,  0.37099522,  0.92983144,  0.76886753]
d = origin_dispersion
#get_histogram(math_wait, dispersion, d)

# желаемое число уровней квантования
# L = 2**R
# R=log2(L)
L = 8

# шаг квантования
# (max(d) - min(d)) / L
#np.round([(i - min(data)) / step for i in data])
print()

def mse(res1, res2):
	e = 0
	for i in range(len(res1)):
		e += (res2[i]-res1[i])**2
	e = (e / (len(res1) - 1))**(1/2)
	return e

def uniform_quantization(arr, L):
	res = []
	xmin, xmax, N = min(arr), max(arr), len(arr)
	q = (xmax - xmin) / (L-1)
	for j in range(N):
		# arr[j] = arr[j: j+q]
		a = (arr[j] - xmin)/q
		res.append(xmin + (a+0.5)*q)

	E = []
	dictionary_a = {}
	dictionary_r = {}
	for i in range(N):
		if arr[i] in dictionary_a:
			dictionary_a[arr[i]] = dictionary_a.get(arr[i])+1
		else:
			dictionary_a[arr[i]] = 1
		if res[i] in dictionary_r:
			dictionary_r[res[i]] = dictionary_r.get(res[i])+1
		else:
			dictionary_r[res[i]] = 1
		E.append(abs(res[i]-arr[i]))
	ea = 0
	er = 0
	for i in dictionary_a:
		ea += (math.fabs(dictionary_a.get(i) / N) * np.log2(dictionary_a.get(i) / N))
	for i in dictionary_r:
		er += (math.fabs(dictionary_r.get(i) / N) * np.log2(dictionary_r.get(i) / N))
	print('ошибка квантования', (max(E)+min(E))/2)
	print('энтропия', ea, '-', er, '=', ea - er)
	print('среднеквадратическая ошибка квантования msi', q*q/12, mse(arr, res))
	return res

def abs_max_in_block(arr):
	res = []
	for i in range(len(arr)):
		res.append(abs(arr[i]))
	return max(res)

def LM_quantization(arr, L):
	res = []
	xmin, xmax, N = min(arr), max(arr), len(arr)
	q = (xmax - xmin) / (L-1)
	q_steps = []
	for j in range(0, N):
		interval = arr[j : j+L]
		M = abs_max_in_block(interval)
		delta = 2*M / L # локальный ур квантования
		q_steps.append(delta)
		#k = L*delta / 2*M # масштабирование
		#print(interval)
		#print(len(interval), M, delta, k)
		a = round(arr[j] / delta)*delta
		res.append(a)

	E = []
	for i in range(N):
		E.append(abs(res[i]-arr[i]))
	print('ошибка квантования', (max(E)+min(E))/2)
	E = []
	for each in q_steps:
		E.append(each*each /12)
	ea = 0
	er = 0
	dictionary_a = {}
	dictionary_r = {}
	for i in range(N):
		if arr[i] in dictionary_a:
			dictionary_a[arr[i]] = dictionary_a.get(arr[i])+1
		else:
			dictionary_a[arr[i]] = 1
		if res[i] in dictionary_r:
			dictionary_r[res[i]] = dictionary_r.get(res[i])+1
		else:
			dictionary_r[res[i]] = 1
	for i in dictionary_a:
		ea += (math.fabs(dictionary_a.get(i) / N) * np.log2(dictionary_a.get(i) / N))
	for i in dictionary_r:
		er += (math.fabs(dictionary_r.get(i) / N) * np.log2(dictionary_r.get(i) / N))
	print('энтропия', ea - er, ea, er)
	print('среднеквадратическая ошибка квантования msi', (max(E)+min(E))/2, mse(arr, res))
	return res

print('равномерное квантование')
print(uniform_quantization(d, L))
print('неравномерное квантование ллойда-макса')
print(LM_quantization(d,L))
print('оригинальный массив')
print(d)

def centroid_recalc(arr, book_size = 1):
	res = []
	xmin, xmax, N = min(arr), max(arr), len(arr)
	# вычисление центроида
	centroid = 1/ N * sum(arr)
	# centroid = mean(arr)
	# centroid = scipy.ndimage.measurements.center_of_mass(arr)
	# рассеивание
	dispersion = 0
	for x in arr:
		dispersion += x - centroid
	dispersion = dispersion / N
	for i in arr:
		res.append()
	return centroid, dispersion, res

# distanve = sqrt( sum(x1_i - x2_i)^2)
def eucidean_distance(arr1, arr2):
	distance = 0.0
	for i in range(len(arr1)-1):
		distance += (arr1[i]-arr2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def get_best_mathing_unit(codebooks, test_arr):
	distances = []
	for codebook in codebooks:
		dist = eucidean_distance(codebook, test_arr)
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]

# Create a random codebook vector
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

def LindeBuzoGrey(array):
	book_size = 1
	N = len(array)
	while book_size < L:
		parts =  
		book = random_codebook(array, book_size, N)
		for i in range(len())
		centroids = []
		distorsion = []
		for arr in parts:
			centroid, dispersion, res = centroid_recalc(arr)
			centroids.append(centroid)
			distorsion.append(dispersion)
		book_size = 2*book_size
	distorsion = mean(distorsion)

def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]

def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)
	return(predictions)

# https://machinelearningmastery.com/implement-learning-vector-quantization-scratch-python/


'''
	for j in range(0, N-1, L-1):
		i = ((( j+L if j+L < N else N-1)))
		print(j, i)
		print(arr[j:i])'''