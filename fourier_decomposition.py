"""
Harmonic decomposition of intensity maps and velocity fields 

...
Functions for fitting the best ellipse to image data by minimising the dominant
harmonic coefficients, and subsequently extracting the harmonic decomposition




Adapted from the IDL code Kinemetry. 
IF YOU USE THIS CODE TO PERFORM HARMONIC DECOMPOSITION OF DATA 
PLEASE CITE Krajnovic et al. 2006, MNRAS 336, 787; THE ORIGINAL DEVELOPERS OF THE METHOD


...

Author
Adam B. Watts; October 2019
International Centre for Radio Astronomy Research
The University of Western Australia
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d, griddata
from scipy import ndimage
from astropy.table import Table
from lmfit import Minimizer, Parameters, report_fit

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning


def main():

	# create = False
	# if create == True:

	# 	Imap, Vfield, R_opt = model_intensity_velocity_map_2comp(PA = [0,0])
	# 	xcoord = []
	# 	ycoord = []
	# 	Imap_signal = []
	# 	Vfield_signal = []
	# 	noise = []
	# 	for yy in range(len(Imap)):
	# 		for xx in range(len(Imap)):
	# 			xcoord.extend([xx])
	# 			ycoord.extend([yy])
	# 			Imap_signal.extend([Imap[yy,xx]])
	# 			Vfield_signal.extend([Vfield[yy,xx]])
	# 			noise.extend([1])

	# 	binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
	# 		np.array(xcoord), np.array(ycoord), np.array(Imap_signal), np.array(noise),  20., plot=0, quiet=0, pixelsize=1)

	# 	Imap_signal = np.array(Imap_signal)
	# 	Vfield_signal = np.array(Vfield_signal)

	# 	Nbins = np.nanmax(binNum)
	# 	Imap_binvals = np.zeros(Nbins)
	# 	Vfield_binvals = np.zeros(Nbins)

	# 	for bb in range(Nbins):
	# 		pixinbin = np.where(binNum == bb)[0]
	# 		Imap_binvals[bb] = np.median(Imap_signal[pixinbin])
	# 		Vfield_binvals[bb] = np.median(Vfield_signal[pixinbin])

		
	# 	dens2 = (Imap_signal/np.array(noise))**4
	# 	mass = ndimage.sum(dens2, labels=binNum, index=range(Nbins))
	# 	xNodes = ndimage.sum(xcoord*dens2, labels=binNum, index=range(Nbins))/mass
	# 	yNodes = ndimage.sum(ycoord*dens2, labels=binNum, index=range(Nbins))/mass

	# 	np.savetxt('./data/voroni_bins_PA0_1comp.txt', np.column_stack([np.array(xcoord), np.array(ycoord), binNum]),
	# 				fmt=b'%10.6f %10.6f %8i')

	# 	np.savetxt('./data/voroni_bins_values_PA0_1comp.txt', np.column_stack([xNode, yNode, Imap_binvals, Vfield_binvals]),
	# 				fmt=b'%10.6f %10.6f %10.6f %10.6f')
		

	# 	# Imap_pix_binvals = np.zeros(len(binNum))
	# 	# Vfield_pix_binvals = np.zeros(len(binNum))
	# 	# for bb in range(Nbins):
	# 		pixinbin = np.where(binNum == bb)[0]
	# 		# Imap_pix_binvals[pixinbin] = np.median(Imap_signal[pixinbin])
	# 		# Vfield_pix_binvals[pixinbin] = np.median(Vfield_signal[pixinbin])


	create_asym = False
	if create_asym == True:
		Imap, Vfield, R_opt = model_1comp_asymmetric()

		plt.imshow(Imap)
		plt.show()
		plt.imshow(Vfield)
		plt.show()

		xcoord = []
		ycoord = []
		Imap_signal = []
		Vfield_signal = []
		noise = []
		for yy in range(len(Imap)):
			for xx in range(len(Imap)):
				xcoord.extend([xx])
				ycoord.extend([yy])
				Imap_signal.extend([Imap[yy,xx]])
				Vfield_signal.extend([Vfield[yy,xx]])
				noise.extend([1])

		binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
			np.array(xcoord), np.array(ycoord), np.array(Imap_signal), np.array(noise),  30., plot=0, quiet=0, pixelsize=1)

		Imap_signal = np.array(Imap_signal)
		Vfield_signal = np.array(Vfield_signal)

		Nbins = np.nanmax(binNum)
		Imap_binvals = np.zeros(Nbins)
		Vfield_binvals = np.zeros(Nbins)

		for bb in range(Nbins):
			pixinbin = np.where(binNum == bb)[0]
			Imap_binvals[bb] = np.median(Imap_signal[pixinbin])
			Vfield_binvals[bb] = np.median(Vfield_signal[pixinbin])

		
		dens2 = (Imap_signal/np.array(noise))**4
		mass = ndimage.sum(dens2, labels=binNum, index=range(Nbins))
		xNodes = ndimage.sum(xcoord*dens2, labels=binNum, index=range(Nbins))/mass
		yNodes = ndimage.sum(ycoord*dens2, labels=binNum, index=range(Nbins))/mass

		# plt.scatter(xNodes,xNode)
		# plt.show()

		np.savetxt('./data/voroni_bins_HIasym.txt', np.column_stack([np.array(xcoord), np.array(ycoord), binNum]),
					fmt=b'%10.6f %10.6f %8i')

		np.savetxt('./data/voroni_bins_values_HIasym.txt', np.column_stack([xNodes, yNodes, Imap_binvals, Vfield_binvals]),
					fmt=b'%10.6f %10.6f %10.6f %10.6f')


	
	data = np.loadtxt('./data/voroni_bins_values_HIasym.txt')
	data = np.array([data[:,0],data[:,1],data[:,3]]).T
	params, coeffs = harmonic_decomposition(
						data, moment = 1, Vsys = False, centre = [250.5,250.5], PAQ = [0,0.766], LOUD = True)


	plot_coeffs_radial(params,coeffs)


	





def harmonic_decomposition(data, moment = 0, pix_scale = None, centre = [0,0], PAQ = False, image = False, Vsys = True, LOUD = False):
	"""
	data should be input as [xNodes, yNodes, binValues], o a 2D image array
	"""


	if image == True:									#convert 2D image to image array
		xdim = np.arange(len(data[0,:]))
		ydim = np.arange(len(data[:,0]))
		xNodes, yNodes = np.meshgrid(xdim, ydim)
		data = np.array([xNodes.flatten(), yNodes.flatten(), data.flatten()]).T

	if pix_scale != None:								#data coordinates are already in pixels
		data[:,0:2] /= pix_scale

	x0 = centre[0]										#estimates of image centre, default 0,0
	y0 = centre[1]

	sample_radii = np.arange(5,201) + 1.1**(np.arange(5,201))
	if image == True:
		sample_radii = sample_radii[sample_radii < 0.5*np.max(xdim)-10]
	# sample_radii = [240,244,248]
	
	best_ellipse_params = []
	harmonic_coeffs = []
	harmonic_koeffs = []

	for rr in range(len(sample_radii)):
		radius = sample_radii[rr]
		print(radius)
		ellipse_params = calc_best_ellipse(data, radius, x0, y0, moment = moment, 
											Vsys = Vsys, PAQ = PAQ, LOUD=LOUD)							#get best fitting ellipse parameters

		coeffs = ellipse_harmonic_expansion(ellipse_params, data, show=False)							#extract coefficients along ellipse
		print(coeffs)
		if coeffs == None:																				#signifies ellipses are going outside range of data
			break
		else:
			ellipse_params['PA'] += 90.e0 																#PA measured East of North on sky 
			if pix_scale != None:
				ellipse_params['R'] *= pix_scale														#convert to actual radius
			best_ellipse_params.append([ellipse_params[key] for key in ['R','PA','q','x0','y0']])
			harmonic_coeffs.append([coeffs[c] for c in 
									['A0','A05','B05','A1','B1','A2','B2','A3','B3','A4','B4','A5','B5']])

	best_ellipse_params = np.array(best_ellipse_params)		
	harmonic_coeffs = np.array(harmonic_coeffs)		

	return best_ellipse_params, harmonic_coeffs

def calc_best_ellipse(data, radius, x0, y0, moment = 0, Vsys = True, PAQ = False, LOUD = False):

	if Vsys == True:
		Vsys = 1
	else:
		Vsys = 0

	if PAQ == False:																				#get rough best PA & q for lmfit

		PA_range = np.linspace(-95.,95,41)
		q_range = np.linspace(0.2,1.0,24)

		ellipse_params = {'R':radius, 'PA':0, 'q':0,'x0':x0,'y0':y0,'Vsys':Vsys,'moment':moment,'order':3}		#initialise parameters
		min_chisq = 1.e11																			#BIIG

		for PA in PA_range:																			#rough chi-sq gridding for input to lmfit
			for q in q_range:
				# print(PA,q)
				ellipse_params['PA'] = PA
				ellipse_params['q'] = q
				chisq = ellipse_params_fitfunc(ellipse_params, data, LM = False)
				if chisq < min_chisq:
					PA_min = PA
					q_min = q
					min_chisq = chisq
		if LOUD == True:
			print('Rough fit parameters')															#best rough fit parameters
			print('R = ', radius)
			print('PA = ', PA_min)
			print('q = ', q_min)	
		ellipse_params = Parameters()																#set up lmfit parameters and limits
		ellipse_params.add('R', value = radius, vary=False)
		ellipse_params.add('PA', value = PA_min, min = -95, max=95)
		ellipse_params.add('q', value = q_min, min=0.2, max=1)
		ellipse_params.add('x0', value = x0)
		ellipse_params.add('y0', value = y0)
		ellipse_params.add('moment', value = moment, vary = False)
		ellipse_params.add('order',value = 3, vary = False)
		
		if moment == 0 or moment == 1:																				#centre fitting for odd moments is currently 3 parameters
			ellipse_params['x0'].set(vary=False)													#to fit 4 variables. the paper lied. 
			ellipse_params['y0'].set(vary=False)													#fix the centre

		mini = Minimizer(ellipse_params_fitfunc, ellipse_params,									#lmfit minimizer
						fcn_args = (data,), fcn_kws = {'LM':True})
		fit_kws = {'ftol':1.e-9,'xtol':1.e-9}
		result = mini.minimize(method='leastsq',**fit_kws)											#optimise parameters
		if LOUD == True:
			report_fit(result)

		params = result.params.valuesdict()															#get results as a parameter dictionary
		params['order'] = 5																			#set to extract higher order moments now the best elipse is found								
	else:																							#fix PA and q
		params = {'R':radius, 'PA':PAQ[0], 'q':PAQ[1], 'x0':x0, 'y0':y0, 'Vsys':Vsys, 'moment':moment, 'order':5}		
																												
	return params

def ellipse_params_fitfunc(ellipse_params, data, LM = True):
	if LM == True:
		ellipse_params = ellipse_params.valuesdict()								#convert lmfit parameters to a python dictionary
	fit_params = ellipse_harmonic_expansion(ellipse_params, data)

	if fit_params == None:															#returns large chi-sq or large minimizing coeffs
		fit_params = {'A0':1.e5,'A1':1.e5,'B1':1.e5,'A2':1.e5,'B2':1.e5,'A3':1.e5,'B3':1.e5}																
		
	if ellipse_params['moment'] == 0:
		coeffs = np.array([fit_params[c] for c in ['A1','B1','A2','B2']])			#coefficients to minimze for even moments
	elif ellipse_params['moment'] == 1:
		coeffs = np.array([fit_params[c] for c in ['A1','A3','B3']])				#coefficients to minimize for odd moments
	
	if LM == False:
		coeffs = np.nansum(coeffs * coeffs)											#chi-sq for rough grid 
															
	return coeffs																	#return chisq, or coeffs to lmfit 

def ellipse_harmonic_expansion(params, data, show = False):
	phi, samples = 	sample_ellipse(params, data, show = show)
	if phi == None:									 									#Indicates there is too much data missing from the ellipse 
		fit_params = None	
	else:
		kwargs = {'ftol':1.e-9,'xtol':1.e-9}
		if params['order'] == 3:

			if params['moment'] == 0:													#even moment maps
				fit, covar = curve_fit(harmonic_expansion_even, phi, samples,**kwargs)	#fit 
				fit_params = {'A0':fit[0],'A1':fit[1],'B1':fit[2],
						'A2':fit[3],'B2':fit[4]}

			elif params['moment'] == 1:													#odd moment maps
				if params['Vsys'] == 1:
					fit, covar = curve_fit(harmonic_expansion_odd_Vsys, phi, samples,**kwargs)	#fit 
					fit_params = {'A0':fit[0],'A1':fit[1],'B1':fit[2],
							'A3':fit[3],'B3':fit[4]}
				else:
					fit, covar = curve_fit(harmonic_expansion_odd, phi, samples,**kwargs)	#fit 
					fit_params = {'A0':0,'A1':fit[0],'B1':fit[1],
							'A3':fit[2],'B3':fit[3]}
		if params['order'] == 5:
			if params['Vsys'] == 1:
				fit, covar = curve_fit(harmonic_expansion_O5_Vsys, phi, samples,**kwargs)
				fit_params = {'A0':fit[0],'A05':fit[1],'B05':fit[2],'A1':fit[3],'B1':fit[4],
						'A2':fit[5],'B2':fit[6],'A3':fit[7],'B3':fit[8],
						'A4':fit[9],'B4':fit[10],'A5':fit[11],'B5':fit[12]}
			else:		
				fit, covar = curve_fit(harmonic_expansion_O5, phi, samples,**kwargs)
				fit_params = {'A0':0,'A05':fit[0],'B05':fit[1],'A1':fit[2],'B1':fit[3],
						'A2':fit[4],'B2':fit[5],'A3':fit[6],'B3':fit[7],
						'A4':fit[8],'B4':fit[9],'A5':fit[10],'B5':fit[11]}
				
	return fit_params
	
def sample_ellipse(params, data, show = False):
	PA = params['PA'] * np.pi / 180.e0 												#apparently functions return dict's changed even if you don't return the dict, this ensures params['PA'] remains in degrees
	Nsamp = len(np.arange(20*params['R'])[np.arange(20*params['R']) < 100])			#minimum 20 samples, decreases at lower radii. don't ask, is in IDL code
	phi = np.linspace(0., 2.e0*np.pi, Nsamp) 
	
	xgal = params['R'] * np.cos(phi)												#galactocentric coordinates
	ygal = params['R'] * np.sin(phi) * params['q']

	xsky = params['x0'] + xgal * np.cos(PA) - ygal * np.sin(PA)						#coorindates on image
	ysky = params['y0'] + xgal * np.sin(PA) + ygal * np.cos(PA)

	if params['moment'] == 0:
		samples = np.zeros(Nsamp)
		for ss in range(Nsamp):
			kernel = np.where((data[:,0] < xsky[ss] + 3) & (data[:,0] > xsky[ss] - 3) &
							(data[:,1] < ysky[ss] + 3) & (data[:,1] > ysky[ss] - 3))[0]
			interp_x = data[kernel,0]
			interp_y = data[kernel,1]
			interp_data = data[kernel,2]
			samples[ss] = griddata(np.array([interp_x,interp_y]).T,interp_data,np.array([xsky[ss],ysky[ss]]))


	if params['moment'] == 1:
		sample_coords = np.array([xsky,ysky]).T
		samples = griddata(data[:,0:2], data[:,2], sample_coords)						#interpolate. consistent with IDL griddata usage

	if show == True:
		fig = plt.figure(figsize=(16,8))
		gs = gridspec.GridSpec(1,2) 
		ellipse_ax = fig.add_subplot(gs[0,0])
		sample_ax = fig.add_subplot(gs[0,1])

		xnodes = data[:,0]
		ynodes = data[:,1]
		bin_values = data[:,2]
		ellipse_ax.scatter(xnodes,ynodes, c=bin_values)
		ellipse_ax.scatter(xsky,ysky,s=1,color='Black')

		sample_ax.plot(phi,samples, color='Black')
		sample_ax.set_xlabel('Azimuth $\phi$',fontsize=15)
		sample_ax.set_ylabel('Intensity',fontsize=15)
		# sample_ax.set_ylim([0.2,0.6])
		plt.show()

	good = np.where(np.isnan(samples) == False)[0]
	if len(good) < 0.75 * Nsamp:													#need at least 3/4 of the ellipse to be sampled
		phi = None
		samples = None
	else:
		phi = phi[good].tolist()
		samples = samples[good].tolist()

	return phi, samples

def harmonic_expansion_even(phi, A0, A1, B1, A2, B2):
	#harmonic expansion for minimzing parameters for even moments
	H = A0 + A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A2 * np.sin(2.e0*phi) + B2 * np.cos(2.e0*phi)
	return H

def harmonic_expansion_odd_Vsys(phi, A0, A1, B1, A3, B3):
	#harmonic expansion for minimzing parameters for odd moments
	H = A0 + A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A3 * np.sin(3.e0*phi) + B3 * np.cos(3.e0*phi)
	return H

def harmonic_expansion_odd(phi, A1, B1, A3, B3):
	#harmonic expansion for minimzing parameters for odd moments
	H = A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A3 * np.sin(3.e0*phi) + B3 * np.cos(3.e0*phi)
	return H

def harmonic_expansion_O5_Vsys(phi, A0, A05, B05, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5):
	#full harmonic expansion for best-fit elllipse
	H = A0 + \
		A05 * np.sin(0.5*phi) + B05 * np.cos(0.5*phi) +\
		A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A2 * np.sin(2.e0*phi) + B2 * np.cos(2.e0*phi) + \
		A3 * np.sin(3.e0*phi) + B3 * np.cos(3.e0*phi) + \
		A4 * np.sin(4.e0*phi) + B4 * np.cos(4.e0*phi) + \
		A5 * np.sin(0.5e0*phi) + B5 * np.cos(0.5e0*phi)
	return H

def harmonic_expansion_O5(phi,A05, B05, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5):
	#full harmonic expansion for best-fit elllipse
	H = A05 * np.sin(0.5*phi) + B05 * np.cos(0.5*phi) +\
		A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A2 * np.sin(2.e0*phi) + B2 * np.cos(2.e0*phi) + \
		A3 * np.sin(3.e0*phi) + B3 * np.cos(3.e0*phi) + \
		A4 * np.sin(4.e0*phi) + B4 * np.cos(4.e0*phi) + \
		A5 * np.sin(5.e0*phi) + B5 * np.cos(5.e0*phi)
	return H

# def extract_aperature():


## plotting ##

def plot_PA_q_koeffs(best_ellipse_params, harmonic_coeffs, moment):

	A0 = harmonic_coeffs[:,0]
	K1 = np.sqrt( harmonic_coeffs[:,1]**2.e0 + harmonic_coeffs[:,2]**2.e0 )
	K2 = np.sqrt( harmonic_coeffs[:,3]**2.e0 + harmonic_coeffs[:,4]**2.e0 )
	K3 = np.sqrt( harmonic_coeffs[:,5]**2.e0 + harmonic_coeffs[:,6]**2.e0 )
	K4 = np.sqrt( harmonic_coeffs[:,7]**2.e0 + harmonic_coeffs[:,8]**2.e0 )
	K5 = np.sqrt( harmonic_coeffs[:,9]**2.e0 + harmonic_coeffs[:,10]**2.e0 )


	fig = plt.figure(figsize=(7,18))
	gs = gridspec.GridSpec(4,1, hspace = 0) 
	PA_ax = fig.add_subplot(gs[0,0])
	q_ax = fig.add_subplot(gs[1,0])
	K1_ax = fig.add_subplot(gs[2,0])
	K5_ax = fig.add_subplot(gs[3,0])

	PA_ax.set_ylim([45, 135])
	q_ax.set_ylim([0, 1.1])

	PA_ax.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	q_ax.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	K1_ax.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)

	PA_ax.plot(best_ellipse_params[:,0], best_ellipse_params[:,1])
	q_ax.plot(best_ellipse_params[:,0], best_ellipse_params[:,2])

	PA_ax.set_title('Asym Vel, centre brightest pixel fit PAQ')

	if moment == 0:
		K1_ax.plot(best_ellipse_params[:,0], A0)
		K5_ax.plot(best_ellipse_params[:,0], K1/A0)

		K1_ax.set_ylim([0.05,1])
		K5_ax.set_ylim([0, 1])



	if moment == 1:
		K1_ax.plot(best_ellipse_params[:,0], K1)
		K5_ax.plot(best_ellipse_params[:,0], K4/K1)

		K1_ax.set_ylim([0,300])
		K5_ax.set_ylim([0, 0.15])


	plt.show()

def plot_ellipse(params, axes = None):
	PA = (params[1] - 90) * np.pi / 180.e0 												#apparently functions return dict's changed even if you don't return the dict, this ensures params['PA'] remains in degrees
	phi = np.linspace(0., 2.e0*np.pi, 100) 
	
	xgal = params[0] * np.cos(phi)												#galactocentric coordinates
	ygal = params[0] * np.sin(phi) * params[2]

	xsky = params[3] + xgal * np.cos(PA) - ygal * np.sin(PA)						#coorindates on image
	ysky = params[4] + xgal * np.sin(PA) + ygal * np.cos(PA)

	if axes != None:
		axes.scatter(xsky,ysky,color='Black',s=1)
	else:
		plt.scatter(xsky,ysky,color='Black',s=1)

def plot_all(mom0_data, mom1_data, params_1, params_2, coeffs_1, coeffs_2, vel_bins, spectrum_1,spectrum_2):

	A0_1 = coeffs_1[:,0]
	K1_1 = np.sqrt( coeffs_1[:,1]**2.e0 + coeffs_1[:,2]**2.e0 )
	K2_1 = np.sqrt( coeffs_1[:,3]**2.e0 + coeffs_1[:,4]**2.e0 )
	K3_1 = np.sqrt( coeffs_1[:,5]**2.e0 + coeffs_1[:,6]**2.e0 )
	K4_1 = np.sqrt( coeffs_1[:,7]**2.e0 + coeffs_1[:,8]**2.e0 )
	K5_1 = np.sqrt( coeffs_1[:,9]**2.e0 + coeffs_1[:,10]**2.e0 )

	A0_2 = coeffs_2[:,0]
	K1_2 = np.sqrt( coeffs_2[:,1]**2.e0 + coeffs_2[:,2]**2.e0 )
	K2_2 = np.sqrt( coeffs_2[:,3]**2.e0 + coeffs_2[:,4]**2.e0 )
	K3_2 = np.sqrt( coeffs_2[:,5]**2.e0 + coeffs_2[:,6]**2.e0 )
	K4_2 = np.sqrt( coeffs_2[:,7]**2.e0 + coeffs_2[:,8]**2.e0 )
	K5_2 = np.sqrt( coeffs_2[:,9]**2.e0 + coeffs_2[:,10]**2.e0 )


	fig = plt.figure(figsize=(15,20))
	gs = gridspec.GridSpec(7,2, hspace = 0.22, left = 0.05,right=0.99,top=0.99,bottom=0.03,wspace=0.15) 
	HI_ax = fig.add_subplot(gs[0:2,0])
	PA_ax_1 = fig.add_subplot(gs[2,0])
	q_ax_1 = fig.add_subplot(gs[3,0],sharex = PA_ax_1)
	K1_ax_1 = fig.add_subplot(gs[4,0],sharex = PA_ax_1)
	K5_ax_1 = fig.add_subplot(gs[5,0],sharex = PA_ax_1)
	spec_ax_1 = fig.add_subplot(gs[6,0])

	Vel_ax = fig.add_subplot(gs[0:2,1])
	PA_ax_2 = fig.add_subplot(gs[2,1],sharex = PA_ax_1, sharey = PA_ax_1)
	q_ax_2 = fig.add_subplot(gs[3,1],sharex = PA_ax_1, sharey = q_ax_1)
	K1_ax_2 = fig.add_subplot(gs[4,1],sharex = PA_ax_1)
	K5_ax_2 = fig.add_subplot(gs[5,1],sharex = PA_ax_1)
	spec_ax_2 = fig.add_subplot(gs[6,1],sharex = spec_ax_1, sharey = spec_ax_1)

	PA_ax_1.set_ylim([45, 135])
	q_ax_1.set_ylim([0, 1.1])

	HI_ax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 0,top=False,bottom=False,left=False,right=False)
	Vel_ax.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 0,top=False,bottom=False,left=False,right=False)


	# PA_ax_1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	# q_ax_1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	# K1_ax_1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)

	# PA_ax_2.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	# q_ax_2.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	# K1_ax_2.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)

	Vel_ax.set_aspect('equal')

	HImap = HI_ax.imshow(mom0_data)
	for ii in range(len(params_1)-10,len(params_1)):
		plot_ellipse(params_1[ii,:], axes = HI_ax)

	velmap = Vel_ax.scatter(mom1_data[:,0],mom1_data[:,1],c=mom1_data[:,2])
	for ii in range(len(params_2)-10,len(params_2)):
		plot_ellipse(params_2[ii,:], axes = Vel_ax)

	PA_ax_1.plot(params_1[:,0], params_1[:,1])
	q_ax_1.plot(params_1[:,0], params_1[:,2])
	PA_ax_2.plot(params_1[:,0], params_1[:,1])
	q_ax_2.plot(params_1[:,0], params_1[:,2])


	K1_ax_1.plot(params_1[:,0], A0_1)
	K5_ax_1.plot(params_1[:,0], K1_1/A0_1)
	K1_ax_1.set_ylim([0.05,1])
	K5_ax_1.set_ylim([0, 1])

	K1_ax_2.plot(params_2[:,0], K1_2)
	K5_ax_2.plot(params_2[:,0],K5_2/K1_2)

	K1_ax_2.set_ylim([0,200])
	K5_ax_2.set_ylim([0, 0.15])


	spec_ax_1.plot(vel_bins,spectrum_1)
	spec_ax_2.plot(vel_bins,spectrum_2)

	PA_ax_1.set_ylabel('PA [deg]')
	PA_ax_2.set_ylabel('PA [deg]')

	q_ax_1.set_ylabel('q [cos(i)]')
	q_ax_2.set_ylabel('q [cos(i)]')

	K1_ax_1.set_ylabel('A0 : Surface brightness')
	K1_ax_2.set_ylabel('K1 : Circular Velocity')

	K5_ax_1.set_ylabel('K1/A0 : m = 1 amplutude')
	K5_ax_2.set_ylabel('K5/K1 : higher order rotation')

	spec_ax_1.set_ylabel('HI mass')
	spec_ax_2.set_ylabel('HI mass')
	spec_ax_1.set_xlabel('LOS velocity')
	spec_ax_2.set_xlabel('LOS velocity')


	K5_ax_1.set_xlabel('Radius [pix]')
	K5_ax_2.set_xlabel('Radius [pix]')

	# plt.tight_layout()
	fig.savefig('./data/test.png',dpi=200)


def plot_coeffs_radial(params, coeffs):
	
	radius = params[:,0]
	

	fig = plt.figure(figsize=(15,8))
	gs = gridspec.GridSpec(8,4, hspace = 0.0, left = 0.05, right=0.99, top=0.99, bottom=0.0, wspace=0.25) 
	
	A0_ax = fig.add_subplot(gs[0,0])
	A05_ax = fig.add_subplot(gs[1,0])
	A1_ax = fig.add_subplot(gs[2,0])
	A2_ax = fig.add_subplot(gs[3,0])
	A3_ax = fig.add_subplot(gs[4,0])
	A4_ax = fig.add_subplot(gs[5,0])
	A5_ax = fig.add_subplot(gs[6,0])

	B05_ax = fig.add_subplot(gs[1,1])
	B1_ax = fig.add_subplot(gs[2,1])
	B2_ax = fig.add_subplot(gs[3,1])
	B3_ax = fig.add_subplot(gs[4,1])
	B4_ax = fig.add_subplot(gs[5,1])
	B5_ax = fig.add_subplot(gs[6,1])
	
	K05_ax = fig.add_subplot(gs[1,2])
	K1_ax = fig.add_subplot(gs[2,2])
	K2_ax = fig.add_subplot(gs[3,2])
	K3_ax = fig.add_subplot(gs[4,2])
	K4_ax = fig.add_subplot(gs[5,2])
	K5_ax = fig.add_subplot(gs[6,2])

	E05_ax = fig.add_subplot(gs[1,3])
	E1_ax = fig.add_subplot(gs[2,3])
	E2_ax = fig.add_subplot(gs[3,3])
	E3_ax = fig.add_subplot(gs[4,3])
	E4_ax = fig.add_subplot(gs[5,3])
	E5_ax = fig.add_subplot(gs[6,3])


	# RC_in_ax = fig.add_subplot(gs[0,2],sharex = A0_ax)


	# def polyex_RC(radius, costheta, V0, scalePE, R_opt, aa, incl)

	ax_list = [[A05_ax,A1_ax, A2_ax, A3_ax, A4_ax, A5_ax],
				[B05_ax,B1_ax, B2_ax, B3_ax, B4_ax, B5_ax],
				[K05_ax,K1_ax, K2_ax, K3_ax, K4_ax, K5_ax],
				[E05_ax,E1_ax, E2_ax, E3_ax, E4_ax, E5_ax]]

	A0_ax.plot(radius,coeffs[:,0], color = 'Black')
	A0_ax.set_ylabel('A$_0$')		
	A05_ax.set_ylabel('A$_{05}$')		
	B05_ax.set_ylabel('B$_{05}$')		
	K05_ax.set_ylabel('K$_{05}$')		
	E05_ax.set_ylabel('$\phi_{05}$')		
	for ii in range(6):
		for jj in range(2):
			ax_list[jj][ii].plot(radius,coeffs[:,2*ii + jj+1], color = 'Black')
		ax_list[2][ii].plot(radius,np.sqrt(coeffs[:,2*ii+1]**2.e0 + coeffs[:,2*ii+1+1]**2.e0), color = 'Black')
		ax_list[3][ii].plot(radius,np.arctan(coeffs[:,2*ii+1]/coeffs[:,2*ii+1+1]), color = 'Black')
		if ii >= 1:
			ax_list[0][ii].set_ylabel('A$_{}$'.format(ii))
			ax_list[1][ii].set_ylabel('B$_{}$'.format(ii))
			ax_list[2][ii].set_ylabel('K$_{}$'.format(ii))
			ax_list[3][ii].set_ylabel('$\phi_{}$'.format(ii))


	RC_app = polyex_RC(radius,1,300,0.18,125,0.002,40)
	RC_rec = polyex_RC(radius,1,200,0.16,125,0.002,40)
	
	K1_ax.plot(radius,RC_app, color='Orange')
	K1_ax.plot(radius,RC_rec, Color='Green')
	K1_ax.plot(radius,0.5 * (RC_app + RC_rec), color='Blue',ls='--')
	# K1_ax.plot(radius,RC_app - RC_rec, color='Green',ls = '--')
	# K2_ax.plot(radius,RC_app - RC_rec, color='Green',ls = '--')
	# K2_ax.plot(radius,np.sqrt(coeffs[:,3]**2.e0 + coeffs[:,4]**2.e0) + np.sqrt(coeffs[:,9]**2.e0 + coeffs[:,10]**2.e0),color='Blue',ls='--')
	K1_ax.plot(radius,coeffs[:,4] - coeffs[:,0] + np.sqrt(coeffs[:,1]**2.e0 + coeffs[:,2]**2.e0) + np.sqrt(coeffs[:,5]**2.e0 + coeffs[:,6]**2.e0) ,color='Magenta',ls='--')
	K1_ax.plot(radius,coeffs[:,4]  + coeffs[:,0] -  np.sqrt(coeffs[:,1]**2.e0 + coeffs[:,2]**2.e0),color='Magenta',ls='--')

	A5_ax.set_xlabel('Radius')
	B5_ax.set_xlabel('Radius')
	K5_ax.set_xlabel('Radius')
	E5_ax.set_xlabel('Radius')


	A0_ax.set_ylim([-20,20])
	A05_ax.set_ylim([-30,30])
	A1_ax.set_ylim([-2,2])
	A2_ax.set_ylim([-2,2])
	A3_ax.set_ylim([-1,1])
	A4_ax.set_ylim([-1,1])
	A5_ax.set_ylim([-1,1])

	B05_ax.set_ylim([-1,1])
	B1_ax.set_ylim([0,250])
	B2_ax.set_ylim([-20,20])
	B3_ax.set_ylim([-1,1])
	B4_ax.set_ylim([-1,1])
	B5_ax.set_ylim([-1,1])
	

	K05_ax.set_ylim([-30,30])
	K1_ax.set_ylim([0,250])
	K2_ax.set_ylim([-20,20])
	K3_ax.set_ylim([-1,1])
	K4_ax.set_ylim([-1,1])
	K5_ax.set_ylim([-1,1])

	E1_ax.set_ylim([-1.8,1.8])
	E2_ax.set_ylim([-1.8,1.8])
	E3_ax.set_ylim([-1.8,1.8])
	E4_ax.set_ylim([-1.8,1.8])
	E5_ax.set_ylim([-1.8,1.8])

	fig.suptitle('Sym velocity, Vsys = fixed @ 0, fit order 0.5',fontsize = 20)
	fig.savefig('./data/coeffs_Symvel_Vsys0_fit0.5.png',dpi=200)
	plt.show()





### examples ###

def HI_vel_asym_example():
	radius, costheta, R_opt = create_arrays(500, 40, 0, limit = False)

	mom0_1 = create_mom0(radius, costheta, R_opt, R_scale = [0.5,1])
	mom1_1 = create_mom1(radius, costheta, 40, -1, R_opt)

	mom0_2 = create_mom0(radius, costheta, R_opt)
	mom1_2 = create_mom1(radius, costheta, 40, -1, R_opt, Vamp = [200,300], R_PE = [0.15,0.18])
	
	vel_bins, spectrum_1 = hi_spectra(mom0_1,mom1_1)
	vel_bins, spectrum_2 = hi_spectra(mom0_2,mom1_2)



	params_1, coeffs_1 = harmonic_decomposition(
						mom0_1, moment = 0, centre = [250,250], PAQ = [0,0.766], image = True, LOUD = True)


	data = np.loadtxt('./data/voroni_bins_values_VELasym.txt')
	data = np.array([data[:,0],data[:,1],data[:,3]]).T
	params_2, coeffs_2 = harmonic_decomposition(
						data, moment = 1, centre = [250,250], PAQ = [0,0.766], LOUD = True)


	pixbins = np.loadtxt('./data/voroni_bins_VELasym.txt')
	binNum = pixbins[:,2]
	mom1_pix_binvals = np.zeros(len(binNum))
	for bb in range(int(np.max(binNum))):
		pixinbin = np.where(binNum == bb)[0]
		mom1_pix_binvals[pixinbin] = np.median(mom1_2.flatten()[pixinbin])

	mom1_data = np.array([pixbins[:,0],pixbins[:,1], mom1_pix_binvals]).T

	# plot_PA_q_koeffs(best_ellipse_params, harmonic_coeffs, moment = 1)

	plot_all(mom0_1,mom1_data, params_1, params_2, coeffs_1, coeffs_2, vel_bins, spectrum_1,spectrum_2)

###### mock data #####
def create_arrays(dim, incl, PA, limit=True):
	"""
	Creates 2D arrays of radius and angle for the HI toy model

    Parameters
    ----------
    dim : int 	[pixels]
        Dimension N 
	params : list
		List of input parameters
			params[0] = Galaxy inclination 	[deg]
        	
    Returns
    -------
 	radius : N x N array 	[pixels]
 		2D array of galactocentric radii
 	costheta: N x N array
 		2D array of cos(theta) = [-pi, pi] values where theta is the angle counter clockwise from
 		the receding major axis (defined as the positive x-axis)
 	R_opt : float 	[pixels]
 		Value of the optical radius in pixels defined as N/4, making Rmax = 2 R_opt
    """

	radius = np.zeros([dim, dim])
	costheta = np.zeros([dim, dim])
	incl = 1.e0 / np.cos(incl * np.pi / 180.e0)				#inclination correction goes as 1/cos
	PA *= np.pi / 180.e0
	for yy in range(dim):
		for xx in range(dim):

			xsky = ((xx + 1.e0) - 0.5e0 * (dim + 1))
			ysky = ((yy + 1.e0) - 0.5e0 * (dim + 1)) #y coordinate is projected by inclination
			
			xgal = xsky * np.cos(PA) + ysky * np.sin(PA)
			ygal = (-1.e0 *xsky * np.sin(PA) + ysky * np.cos(PA)) * incl

			# th[yy,xx] = theta
			theta = 1
			rad = np.sqrt( (xgal)**2.e0 + ((ygal)**2.e0) )	
			if limit ==  True:
				if rad <= 0.5e0 * (dim + 1.e0):
					radius[yy, xx] = rad
					if xgal != 0:
						costheta[yy, xx] = (np.sign(xgal) *
							np.cos(np.arctan(ygal / xgal)) )
					else:
						costheta[yy, xx] = (np.sign(xgal) *
							np.cos(np.sign(ygal) * np.pi * 0.5e0) )

				else:
					radius[yy, xx] = float('nan')							#no data outside galaxy radius
					costheta[yy, xx] = float('nan')							#further routines will conserve NaN
			elif limit == False:
				radius[yy, xx] = rad
				if xgal != 0:
					costheta[yy, xx] = (np.sign(xgal) *
						np.cos(np.arctan(ygal / xgal)) )
				else:
					costheta[yy, xx] = (np.sign(xgal) *
						np.cos(np.sign(ygal) * np.pi * 0.5e0) )
	R_opt = dim / 4.e0														#define image to cover 2 optical radii						
	return radius, costheta, R_opt

def create_mom0(radius, costheta, R_opt, R_scale = [1,1]):
	"""
	Generates a 2D HI mass map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    params : list
    	List of model input parameters
    		params[0] = Galaxy inclination 	[deg]
	    	params[1] = Model type
	    	params[2] = Asymmetry flag
	    	params[3] = Total HI mass 		[Msun]
	    	params[4:7] = Receding side / symmetric input parameters
	    	params[8:11] = Approaching side parameters
	R_opt : float 	[pixels]
    	Optical radius 

    Returns
    -------
	mom0_map : N x N array 	[Msun/pc^2]
		2D array of projected HI surface densities
	rad1d : array 	[1 / R_opt]
		Radii bins for measured radial HI surface densities
	input_profile : 2 element list of arrays 	[Msun/pc^2]
		Radial projected HI surface density profiles of 
		receding and approaching side respectively
	"""
	dim  = len(radius)
	mom0_map = np.zeros([dim,dim])
	R_scale = [R*R_opt for R in R_scale]

	R_scale_map = R_scale[1] * (1.e0 + (((R_scale[0] - R_scale[1])/R_scale[1]) * 0.5e0* (costheta + 1.e0)))

	mom0_map = np.exp(-1.e0*radius/R_scale_map)
	

	# Rstep = (0.5e0 * dim) / 50.
	# rad1d = np.arange(0, int((dim) / 2) + 2.e0 * Rstep, Rstep)
	# input_receding = np.arange(len(rad1d) - 1)
	# input_approaching = np.arange(len(rad1d) - 1)

	# radius_temp = np.zeros([dim,dim])									#make approaching side have negative 
	# radius_temp[:, 0:int(dim / 2)] = -1.e0 * radius[:, 0:int(dim / 2)]	#radius to make summing easier
	# radius_temp[:, int(dim / 2)::] = radius[:, int(dim / 2)::]
	# for bb in range(len(rad1d) - 1):
	# 	bin_low = rad1d[bb]
	# 	bin_high  = rad1d[bb + 1]
	# 	inbin_app = mom0_map[(radius_temp <= -1.e0 * bin_low) & (radius_temp >
	# 				 -1.e0 * bin_high)]
	# 	inbin_rec = mom0_map[(radius_temp >= bin_low) & (radius_temp < bin_high)]

	# 	input_approaching[bb] = np.nansum(inbin_app) * incl / len(inbin_app)
	# 	input_receding[bb] = np.nansum(inbin_rec) * incl / len(inbin_rec)		#inclination corrected 
	# rad1d = rad1d[0:-1] / R_opt


	return mom0_map #, rad1d , input_profile

def create_mom1(radius, costheta, incl, rad1d, R_opt, Vamp = [200,200], R_PE = [0.164,0.164], alpha = [0.002,0.002], Vdisp = 0):
	"""
	Generates a 2D gas velocity map for symmetric or asymmetric distribution inputs

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    rad1d : array [1 / R_opt]
    	Radii bins for measuring input rotation curve
    params : list
    	List of model input parameters
    		params[0] = Galaxy inclination 	[deg]
	    	params[12] = Asymmetry flag
	    	params[13:15] = Receding side / symmetric input parameters
	    	params[16:18] = Approaching side parameters
	    	params[19] = Velocity dispersion 	[km/s]
	R_opt : float 	[pixels]
    	Optical radius 

    Returns
    -------
	mom1_map : N x N array 	[km/s]
		2D array of projected gas rotational velcoities
	input_RC : array 	[km/s]
		Projected input rotation curve
	"""

	Vamp_map = Vamp[1] * (1.e0 + (((Vamp[0] - Vamp[1]) / Vamp[1]) 
			* 0.5e0 * (costheta + 1.e0)))
	R_PE_map = R_PE[1] * (1.e0 + (((R_PE[0] - R_PE[1]) / R_PE[1])
			* 0.5e0 * (costheta + 1.e0)))
	alpha_map = alpha[1] * (1.e0 + (((alpha[0] - alpha[1]) / alpha[1]) 
			* 0.5e0 * (costheta + 1.e0)))

	# RC_rec = polyex_RC(rad1d * R_opt, 1.e0, Vamp[0], R_PE[0], R_opt, alpha[0], incl)
	# RC_app = polyex_RC(rad1d * R_opt, 1.e0, Vamp[1], R_PE[1], R_opt, alpha[0], incl)
	# input_RC = [RC_rec, RC_app]

	mom1_map = polyex_RC(radius, costheta, Vamp_map, R_PE_map, R_opt, alpha_map, incl)

	if Vdisp >= 0:								#add velocity dispersion
		mom1_map = np.random.normal(mom1_map, Vdisp)

	return mom1_map#, input_RC

def polyex_RC(radius, costheta, V0, scalePE, R_opt, aa, incl):
	"""
	Creates a 2D projected velocity map using the Polyex rotation curve (RC) defined 
	by Giovanelli & Haynes 2002, and used by Catinella, Giovanelli & Haynes 2006

    Parameters
    ----------
    radius : N x N array 	[pixels]
        2D array of galactocentric radii
    costheta : N x N array
        2D array of cos(theta) values from the receding major axis
    V_0 : float 	[km/s]
    	Amplitude of RC
    scalePE : float 	[1 / R_opt]
    	Scale length of exponential inner RC
    R_opt : float 	[pixels]
    	Optical radius
    aa : float
    	Slope of outer, linear part of RC
    incl : float 	[deg]
    	Galaxy inclination

    Returns
    -------
	mom1 : N x N array 	[km/s]
		2D array of inclination corrected rotational velocity of each pixel
	"""

	incl = np.sin(incl * (np.pi / 180.e0))
	R_PE = 1.e0 / (scalePE * R_opt)											#rotation curve scale length Catinella+06
	mom1 = ( (V0 * (1.e0 - np.exp((-1.e0 * radius) * R_PE)) * 
		(1.e0 + aa * radius * R_PE)) * costheta  * incl )
	return mom1

def model_intensity_velocity_map_2comp(dim = 500, GMfact = [800,1500], r0 = [0.4,1.2], re = [0.56,1.2], weights = [4,1], incl = [60,40], PA = [0,0]):

	radius, costheta_disk, Ropt = create_arrays(dim, incl[0],PA[0], limit=False)
	radius, costheta_bulge, Ropt = create_arrays(dim, incl[1],PA[1], limit=False)

	Vc_disk = costheta_disk * GMfact[0] * np.sqrt(radius) / (radius + r0[0]*Ropt)
	Vc_bulge = costheta_bulge * GMfact[1] * np.sqrt(radius) / (radius + r0[1]*Ropt)

	I_disk = np.exp(-radius/(re[0]*Ropt))
	I_bulge = np.exp(-radius/(4.*re[1]*Ropt))

	Vc = Vc_disk + Vc_bulge

	I = weights[0]*I_disk + weights[1]*I_bulge

	return I_bulge, Vc_bulge, Ropt


def model_1comp_asymmetric(dim = 500, incl = 40, PA = 0):

	radius, costheta, R_opt = create_arrays(dim, incl, PA, limit = False)

	mom0 = create_mom0(radius, costheta, R_opt, R_scale = [0.5,1])
	
	mom1 = create_mom1(radius, costheta, incl, -1, R_opt)

	return mom0*4, mom1, R_opt

def hi_spectra(mom0, mom1):

	vel_bins = np.arange(-300, 300 , 2)
	spectrum = np.zeros(len(vel_bins))

	for vel in range(len(vel_bins) - 1):
		inbin = mom0[(mom1 >= vel_bins[vel]) & (mom1 < vel_bins[vel + 1] )]
		spectrum[vel] = np.nansum(inbin)   

	return vel_bins, spectrum


if __name__ == '__main__':

	main()


