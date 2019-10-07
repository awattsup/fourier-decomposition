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
# import pafit as pf 
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

	Imap, Vfield, R_opt = model_intensity_velocity_map(PA = [0,0])
	print(R_opt)
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

	# binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
	# 	np.array(xcoord), np.array(ycoord), np.array(Imap_signal), np.array(noise),  20., plot=0, quiet=0, pixelsize=1)
	# # plt.tight_layout()
	# # plt.show()

	# np.savetxt('./data/voroni_bins_PA0_1comp.txt', np.column_stack([np.array(xcoord), np.array(ycoord), binNum]),
	# 			fmt=b'%10.6f %10.6f %8i')
	# exit()
	data = np.loadtxt('./data/voroni_bins_PA0_1comp.txt')
	xcoord = data[:,0]
	ycoord = data[:,1]
	binNum = data[:,2].astype(int)


	Imap_signal = np.array(Imap_signal)
	Vfield_signal = np.array(Vfield_signal)

	Nbins = np.nanmax(binNum)

	Imap_binvals = np.zeros(Nbins)
	Vfield_binvals = np.zeros(Nbins)
	Imap_pix_binvals = np.zeros(len(binNum))
	Vfield_pix_binvals = np.zeros(len(binNum))
	for bb in range(Nbins):
		pixinbin = np.where(binNum == bb)[0]
		Imap_binvals[bb] = np.median(Imap_signal[pixinbin])
		Vfield_binvals[bb] = np.median(Vfield_signal[pixinbin])
		Imap_pix_binvals[pixinbin] = np.median(Imap_signal[pixinbin])
		Vfield_pix_binvals[pixinbin] = np.median(Vfield_signal[pixinbin])
	
	# plt.scatter(xcoord,ycoord,c=Vfield_pix_binvals)
	# plt.show()
	# exit()


	dens2 = (Imap_signal/np.array(noise))**4
	mass = ndimage.sum(dens2, labels=binNum, index=range(Nbins))
	xNodes = ndimage.sum(xcoord*dens2, labels=binNum, index=range(Nbins))/mass
	yNodes = ndimage.sum(ycoord*dens2, labels=binNum, index=range(Nbins))/mass


	data = Table.read('/home/awatts/Downloads/kinemetry/NGC2974_SAURON_kinematics.dat',format='ascii')

	xNodes = data['XBIN']
	yNodes = data['YBIN']
	binNum = data['N']
	Vfield_binvals = data['VEL']


	x0 = 0
	y0 = 0

	# plt.scatter(xNodes,yNodes,c=Vfield_binvals)
	# plt.colorbar()
	# plt.show()
	# exit()

	toy_model = 0
	if toy_model:
		radius, costheta, R_opt = create_arrays(500,80,50)
		mom0 = create_mom0(radius, costheta, R_opt, R_scale = [0.5,1.5])
		mom1 = create_mom1(radius, costheta, 80, -1, R_opt,Vamp = [100,300])
		# plt.imshow(mom0)
		# plt.show()
		# plt.imshow(mom1)
		# plt.show()



	PA_q_coeffs = np.zeros([len(sample_radii),13])
	K_coeffs = np.zeros([len(sample_radii), 5])



	fig = plt.figure(figsize=(7,18))
	gs = gridspec.GridSpec(4,1) 
	PA_ax = fig.add_subplot(gs[0,0])
	incl_ax = fig.add_subplot(gs[1,0])
	K1_ax = fig.add_subplot(gs[2,0])
	K5_ax = fig.add_subplot(gs[3,0])

	PA_ax.set_ylim([30, 70])
	incl_ax.set_ylim([0, 1.1])
	K1_ax.set_ylim([0, 250])
	K5_ax.set_ylim([0, 0.13])

	PA_ax.plot(best_ellipse_params[:,0]*arcsec_per_pix, best_ellipse_params[:,1])
	incl_ax.plot(best_ellipse_params[:,0]*arcsec_per_pix, best_ellipse_params[:,2])
	K1_ax.plot(best_ellipse_params[:,0]*arcsec_per_pix, harmonic_koeffs[:,0])
	K5_ax.plot(best_ellipse_params[:,0]*arcsec_per_pix, harmonic_koeffs[:,4]/harmonic_koeffs[:,0])
	plt.show()


def harmonic_decomposition(xNodes, yNodes, binValues, moment, pix_scale = False, centre = [0,0]):

	if pix_scale != False								#data coordinates are already in pixels
		xNodes /= pix_scale
		yNodes /= pix_scale

	x0 = centre[0]										#estimates of image centre, default 0,0
	y0 = centre[0]

	data = np.array([xNodes,yNodes,binValues]).T

	sample_radii = np.arange(201) + 1.1**(np.arange(201))
	
	best_ellipse_params = []
	harmonic_coeffs = []
	harmonic_koeffs = []

	for rr in range(len(sample_radii)):
		radius = sample_radii[rr]
		
		ellipse_params =  calc_best_ellipse(data, radius, x0, y0, moment = 1)			#get best fitting ellipse parameters
		coeffs = ellipse_harmonic_expansion(ellipse_params, data, show = False)			#extract coefficients along ellipse
		if coeffs == 0:																	#signifies ellipses are going outside range of data
			break
		else:
			ellipse_params['PA'] += 90.e0 												#PA measured East of North on sky 
			best_ellipse_params.append([ellipse_params[key] for key in ['R','PA','q','x0','y0']])
			harmonic_coeffs.append([coeffs[c] for c in 
									['A0','A1','B1','A2','B2','A3','B3','A4','B4','A5','B5']])

			k_coeffs = []																#K_i ^2 = A_i ^2 + B_i ^2
			for ii in range(5):
				k_coeffs.extend([np.sqrt(coeffs['A{}'.format(ii+1)] ** 2.e0 + coeffs['B{}'.format(ii+1)]**2.e0 )])
			harmonic_koeffs.append(k_coeffs)

	best_ellipse_params = np.array(best_ellipse_params)		
	harmonic_coeffs = np.array(harmonic_coeffs)		
	harmonic_koeffs = np.array(harmonic_koeffs)	

	return best_ellipse_params, harmoinc_coeffs, harmonic_koeffs	


def calc_best_ellipse(data, radius, x0, y0, moment = 0):

	PA_range = np.linspace(-95.,95,51)
	q_range = np.linspace(0.2,1.0,24)

	ellipse_params = {'R':radius, 'PA':0, 'q':0,'x0':x0,'y0':y0,'moment':moment,'order':3}		#initialise parameters
	min_chisq = 1.e11																			#BIIG

	for PA in PA_range:																			#rough chi-sq gridding for input to lmfit
		for q in q_range:
			ellipse_params['PA'] = PA
			ellipse_params['q'] = q
			chisq = ellipse_params_fitfunc(ellipse_params, data, LM = False)
			if chisq < min_chisq:
				PA_min = PA
				q_min = q
				min_chisq = chisq
	
	# print('R = 'radius, 'rough fit PA = ', PA_min, 'q = ', q_min)								#best rough fit parameters

	ellipse_params = Parameters()																#set up lmfit parameters and limits
	ellipse_params.add('R', value = radius, vary=False)
	ellipse_params.add('PA', value = PA_min, min = -95, max=95)
	ellipse_params.add('q', value = q_min, min=0.2, max=1)
	ellipse_params.add('x0', value = x0)
	ellipse_params.add('y0', value = y0)
	ellipse_params.add('moment', value = moment, vary = False)
	ellipse_params.add('order',value = 3, vary = False)
	
	if moment == 1:																				#centre fitting for odd moments is currently 3 parameters
		ellipse_params['x0'].set(vary=False)													#to fit 4 variables. the paper lied. 
		ellipse_params['y0'].set(vary=False)													#fix the centre

	mini = Minimizer(ellipse_params_fitfunc, ellipse_params,									#lmfit minimizer
					fcn_args = (data,), fcn_kws = {'LM':True})
	fit_kws = {'ftol':1.e-9,'xtol':1.e-9}
	result = mini.minimize(method='leastsq',**fit_kws)											#optimise parameters
	# report_fit(result)

	params = result.params.valuesdict()															#get results as a parameter dictionary
	params['order'] = 5																			#set to extract higher order moments now the best elipse is found

	return params

def ellipse_params_fitfunc(ellipse_params, data, LM = True):
	if LM == True:
		ellipse_params = ellipse_params.valuesdict()							#convert lmfit parameters to a python dictionary
	fit_params = ellipse_harmonic_expansion(ellipse_params, data)

	if ellipse_params['moment'] == 0:
		coeffs = np.array([fit_params[c] for c in ['A1','A2','B1','B2']])		#coefficients to minimze for even moments
	elif ellipse_params['moment'] == 1:
		coeffs = np.array([fit_params[c] for c in ['A1','A3','B3']])			#coefficients to minimize for odd moments
	
	if LM == True:
		return coeffs 															#fit coefficients returned to lmfit
	else:
		return np.nansum(coeffs * coeffs)										#chi - squared for rough grid 


def ellipse_harmonic_expansion(params, data, show = False):
	phi, samples = 	sample_ellipse(params, data, show = show)

	if phi == 0 and params['order'] != 3: 										#Indicates there is too much data missing from the ellipse 
		fit_params = 0															#we dont care for rough gridding, but do for fitting the best ellipse, hence order!=3
	else:
		kwargs = {'ftol':1.e-9,'xtol':1.e-9}
		if params['order'] == 3:
			# init = [np.mean(samples),0.1,np.max(samples),0.1,0.1]

			if params['moment'] == 0:													#even moment maps
				fit, covar = curve_fit(harmonic_expansion_O2, phi, samples,**kwargs)	#fit 
				fit_params = {'A0':fit[0],'A1':fit[1],'B1':fit[2],
						'A2':fit[3],'B2':fit[4]}

			elif params['moment'] == 1:													#odd moment maps
				fit, covar = curve_fit(harmonic_expansion_O3, phi, samples,**kwargs)	#fit 
				fit_params = {'A0':fit[0],'A1':fit[1],'B1':fit[2],
						'A3':fit[3],'B3':fit[4]}

		if params['order'] == 5:
			# init = [np.mean(samples),0.1,np.max(samples),0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
			fit, covar = curve_fit(harmonic_expansion_O5, phi, samples,**kwargs)
			fit_params = {'A0':fit[0],'A1':fit[1],'B1':fit[2],
					'A2':fit[3],'B2':fit[4],'A3':fit[5],'B3':fit[6],
					'A4':fit[3],'B4':fit[4],'A5':fit[5],'B5':fit[6]}
	return fit_params
	

def sample_ellipse(params, data, show = False)
	PA = params['PA'] * np.pi / 180.e0 												#apparently functions return dict's changed even if you don't return the dict, this ensures params['PA'] remains in degrees
	Nsamp = len(np.arange(20*params['R'])[np.arange(20*params['R']) < 100])			#minimum 20 samples, decreases at lower radii. don't ask, is in IDL code
	phi = np.linspace(0., 2.e0*np.pi, Nsamp) 
	
	xgal = params['R'] * np.cos(phi)												#galactocentric coordinates
	ygal = params['R'] * np.sin(phi) * params['q']

	xsky = params['x0'] + xgal * np.cos(PA) - ygal * np.sin(PA)						#coorindates on image
	ysky = params['y0'] + xgal * np.sin(PA) + ygal * np.cos(PA)

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
		plt.show()


	good = [np.isnan(samples) == False]
	if len(good) < 0.75 * Nsamp:													#need at least 3/4 of the ellipse to be sampled
		phi = 0, samples = 0
	else:
		phi = phi[good]
		samples = samples[good]

	return phi, samples

def harmonic_expansion_O2(phi, A0, A1, B1, A2, B2):
	#harmonic expansion for minimzing parameters for even moments
	H = A0 + A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A2 * np.sin(2.e0*phi) + B2 * np.cos(2.e0*phi)
	return H

def harmonic_expansion_O3(phi, A0, A1, B1, A3, B3):
	#harmonic expansion for minimzing parameters for odd moments
	H = A0 + A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A3 * np.sin(3.e0*phi) + B3 * np.cos(3.e0*phi)
	return H

def harmonic_expansion_O5(phi, A0, A1, B1, A2, B2, A3, B3, A4, B4, A5, B5):
	#full harmonic expansion for best-fit elllipse
	H = A0 + A1 * np.sin(phi) + B1 * np.cos(phi) + \
		A2 * np.sin(2.e0*phi) + B2 * np.cos(2.e0*phi) + \
		A3 * np.sin(3.e0*phi) + B3 * np.cos(3.e0*phi) + \
		A4 * np.sin(4.e0*phi) + B4 * np.cos(4.e0*phi) + \
		A5 * np.sin(5.e0*phi) + B5 * np.cos(5.e0*phi)
	return H






#### non-interpolator functions ###

# def sample_ellipse(data, params, show=False):
# 	sample = []
# 	params['PA'] *= np.pi / 180.e0
# 	if show == True:
# 		plt.imshow(data)
	
# 	for phi in range(360):
# 		xgal = params['R'] * np.cos(phi * np.pi / 180.e0)
# 		ygal = params['R'] * params['q'] * np.sin(phi * np.pi / 180.e0)
		
# 		xsky = xgal * np.cos(params['PA']) - ygal * np.sin(params['PA'])
# 		ysky = xgal * np.sin(params['PA']) + ygal * np.cos(params['PA'])
		
# 		x_sample = int(round(xsky + params['x0']))
# 		y_sample = int(round(ysky + params['y0']))
# 		sample.extend([data[y_sample,x_sample]])
	
# 		if show == True:
# 			plt.scatter(x_sample,y_sample,s=1)
# 	if show == True:
# 		plt.show()
# 	return sample

# def ellipse_harmonic_expansion(ellipse_params, data, LM = True):
# 	if LM == True:
# 		ellipse_params = ellipse_params.valuesdict()
# 		# print(ellipse_params)
# 	samples = sample_ellipse(data, ellipse_params)
# 	fit_params = fit_harmonic_expansion(samples)

# 	# if LM == True:
# 	# 	plt.plot(samples)
# 	# 	plt.plot(harmonic_expansion_O3(np.arange(360),fit_params['A0'],fit_params['A1'],fit_params['B1'],fit_params['A2'],
# 	# 					fit_params['B2'],fit_params['A3'],fit_params['B3']))
# 	# 	plt.show()

# 	if LM == False:
# 		chisq = calc_chisq(fit_params, moment = ellipse_params['moment'])
# 		return chisq
# 	else:
# 		# print(fit_params)
# 		if ellipse_params['moment'] == 1:
# 			coeffs = np.array([fit_params[c] for c in ['A1','A2','B2','A3','B3']])**2.e0
# 		elif ellipse_params['moment'] == 0:
# 			coeffs = np.array([fit_params[c] for c in ['A1','A2','B1','B2']])**2.e0
# 		# print(coeffs)
# 		return coeffs

# def ellipse_chisq(ellipse_params, data = None, LM = True):

# 	if LM == True:
# 		ellipse_params = ellipse_params.valuesdict()
# 	samples = sample_ellipse(data,  ellipse_params)
# 	fit_params = fit_harmonic_expansion(samples)
# 	chisq = calc_chisq(fit_params, moment = ellipse_params['moment'])
# 	# print(chisq)
# 	return chisq


# def calc_chisq(fit_params, moment):

# 	if moment == 0:
# 		chisq = fit_params['A1']**2.e0 + fit_params['B1']**2.e0 + fit_params['A2']**2.e0 + \
# 				fit_params['B2']**2.e0
# 	if moment == 1:
# 		chisq = fit_params['A1']**2.e0 + fit_params['A2']**2.e0 + fit_params['B2']**2.e0 + \
# 				fit_params['A3']**2.e0 + fit_params['B3']**2.e0
# 	return chisq


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
		(1.e0 + aa * radius * R_PE)) * costheta * incl )
	return mom1

def model_intensity_velocity_map(dim = 500, GMfact = [800,1500], r0 = [0.4,1.2], re = [0.56,1.2], weights = [4,1], incl = [60,40], PA = [0,0]):

	radius, costheta_disk, Ropt = create_arrays(dim, incl[0],PA[0], limit=False)
	radius, costheta_bulge, Ropt = create_arrays(dim, incl[1],PA[1], limit=False)

	Vc_disk = costheta_disk * GMfact[0] * np.sqrt(radius) / (radius + r0[0]*Ropt)
	Vc_bulge = costheta_bulge * GMfact[1] * np.sqrt(radius) / (radius + r0[1]*Ropt)

	I_disk = np.exp(-radius/(re[0]*Ropt))
	I_bulge = np.exp(-radius/(4.*re[1]*Ropt))

	Vc = Vc_disk + Vc_bulge

	I = weights[0]*I_disk + weights[1]*I_bulge

	return I_bulge, Vc_bulge, Ropt

if __name__ == '__main__':

	main()


