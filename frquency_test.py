import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit


def main():

	phi = np.linspace(0.0001,2,100)*np.pi

	Vprof_noasym = 200*np.cos(phi)

	Vamp = [200,300]

	Vamp_array = Vamp[1] * (1.e0 + (((Vamp[0] - Vamp[1]) / Vamp[1]) 
				* 0.5e0 * (np.cos(phi) + 1.e0)))

	Vprof_asym  = Vamp_array * np.cos(phi)*np.cos(0.766)


	Vprof_app = 200*np.cos(phi)*np.cos(0.766)
	Vprof_rec = 300*np.cos(phi)*np.cos(0.766)
	Vprof_avg = 250 * np.cos(phi)*np.cos(0.766)



	plt.plot(phi,Vprof_app, color='Black',ls = ':')
	plt.plot(phi,Vprof_rec, color='Black', ls = '--')
	plt.plot(phi,Vprof_avg, color = 'Black')
	plt.plot(phi, Vprof_asym, color='Red')
	# plt.plot(phi, Vprof_avg + 24.6*np.sin(0.5*phi) +  18.19*np.cos(2*phi),color = 'Blue', ls = '--')
	# plt.plot(phi, Vprof_avg - 24.6*np.sin(0.5*phi) -  18.19*np.cos(2*phi),color = 'Blue', ls = ':')
	plt.plot(phi, Vprof_avg - 50 + 40*np.abs(np.sin(phi)),color = 'Green', ls = '--')
	plt.show()
	exit()

	plt.plot(phi,Vprof_noasym + 24.6*np.sin(0.5*phi) + 18.2*np.cos(2*phi), ls = '--')

	# plt.plot(phi,Vprof_avg + 50*np.sin(0.5*phi))
	plt.show()
	exit()




	# half_freq = 100*np.sin(0.5*phi) / (phi)
	diff = Vprof_noasym - Vprof_asym

	fft = np.fft.fft(diff)
	freq = np.fft.fftfreq(phi.shape[-1])
	print(fft)
	print(freq)

	# plt.plot(freq,fft.real)
	# plt.plot(freq,fft.imag)

	plt.plot(phi,Vprof_noasym)
	plt.plot(phi,Vprof_asym)
	plt.plot(phi,Vprof_noasym - Vprof_asym)

	fit, covar = curve_fit(func, phi, Vprof_asym - Vprof_noasym)
	print(fit)
	plt.plot(phi,func(phi,fit[0],fit[1],fit[2],fit[3]))

	# plt.plot(phi, Vprof_noasym,ls = '--')
	plt.show()






def func(phi,A,B,C, D):
	f = B*np.cos(A*phi) + D
	return f


if __name__ == '__main__':
	main()




