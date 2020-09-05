from os.path import join
#import fitsio
import numpy as np
from scipy import interpolate
from astropy.io import fits


_spplate_hdunames = ('wave','flux','ivar','andmask','ormask','disp','plugmap','sky')

def spSpec_sdss(mjd, plate, fiber, output):
    #filename='/home/edward/Desktop/EdwardProject/DL_codes/DL_codes/data_download/data_download/dr7/spectro/1d_26/%04d/1d/spSpec-%05d-%04d-%03d.fit' % (plate, mjd, plate, fiber)
    filename='C:/Python_Study/for_edward/DL_codes/DL_codes/dr7/spectro/1d_26/%04d/1d/spSpec-%05d-%04d-%03d.fit' % (plate, mjd, plate, fiber)
    print(filename)
    spspec_fits = fits.open(filename)
    extract_data=spspec_fits[0].data
    #spspec_fits = fitsio.FITS(filename)
    #hdr = spspec_fits[0].read_header()
    #print hdr
    #print hdr
    spspec_data= dict()
    #extract_data=spspec_fits[0].read()
    for thishdu in output:
        if thishdu == 'wave':
           c0 = spspec_fits[0].header['COEFF0']#hdr['coeff0']
           c1 = spspec_fits[0].header['COEFF1']#hdr['coeff1']
           npix = spspec_fits[0].header['NAXIS1']#hdr['naxis1']
        # loglam vector is the same for a given plate
           spspec_data[thishdu] = 10**( c0+c1*np.arange(npix, dtype='d') )

        if thishdu == 'flux':
           spspec_data[thishdu] = extract_data[0,:]

        if thishdu == 'noise':
           spspec_data[thishdu] = extract_data[2,:]

           #spspec_fits.close()
    spspec_fits.close()
    return spspec_data





def spec_boss(mjd, plate, fiberid, output):
    #boss_name_path='/Volumes/work_dir/sdss_dr12/v5_7_0/%04d/' % plate
    #filename='/scratch/sdss_data/sdss_dr12/v5_7_0/%04d/spPlate-%04d-%05d.fits' \

    #filename = '/home/edward/Desktop/EdwardProject/DL_codes/DL_codes/data_simulation/dr7/spectro/1d_26/%04d/1d/spSpec-%05d-%04d-%03d.fit' % (plate, plate, mjd)
    filename='C:/Python_Study/for_edward/DL_codes/DL_codes/dr7/spectro/1d_26/%04d/1d/spSpec-%05d-%04d-%03d.fit' % (plate, mjd, plate, fiber)
    spplate_data = dict()
    for thishdu in output:

     spplate_fits = fits.open(filename)
     hdr = spplate_fits[0].header

     if thishdu == 'wave':
        c0 = hdr['coeff0']
        c1 = hdr['coeff1']
        npix = hdr['naxis1']
        # loglam vector is the same for a given plate
        spplate_data[thishdu] = 10**( c0+c1*np.arange(npix, dtype='d') )
     else:
        index = _spplate_hdunames.index(thishdu)-1
        extract_data = spplate_fits[index].data
        spplate_data[thishdu] = extract_data[fiberid-1,:]
        spplate_fits.close()

    return spplate_data

def spec_norm(wave, flux, z):
    wave=wave/(1+z)

    if z <=1.0 :
        #print 'z 1'
        index_norm= (wave>= 4150) & (wave<= 4250)
        #flux=flux/np.mean(flux[index_norm])
    if (z > 1.0) & (z <=1.8):
        index_norm= (wave>= 3020) & (wave<= 3100)
        #print 'z 2'
        #flux=flux/np.mean(flux[index_norm])
    if (z > 1.8) & (z <=2.8):
        index_norm= (wave>= 2150) & (wave<= 2250)
        #print 'z 3'
        #flux=flux/np.mean(flux[index_norm])
    if (z > 2.8) & (z <=4.8):
        index_norm= (wave>= 1420) & (wave<= 1500)
        #print 'z 4'
    #print np.mean(flux[index_norm])
    flux=flux/np.mean(flux[index_norm])
    #else: print("redshift is out of the range!")

    return flux


def spec_norm_v1(wave, flux, error, z):
    wave=wave/(1+z)

    if z <=1.0 :
        #print 'z 1'
        index_norm= (wave>= 4150) & (wave<= 4250)
        #flux=flux/np.mean(flux[index_norm])
    if (z > 1.0) & (z <=1.8):
        index_norm= (wave>= 3020) & (wave<= 3100)
        #print 'z 2'
        #flux=flux/np.mean(flux[index_norm])
    if (z > 1.8) & (z <=2.8):
        index_norm= (wave>= 2150) & (wave<= 2250)
        #print 'z 3'
        #flux=flux/np.mean(flux[index_norm])
    if (z > 2.8) & (z <=4.8):
        index_norm= (wave>= 1420) & (wave<= 1500)
        #print 'z 4'
    #print np.mean(flux[index_norm])
    norm_f = np.mean(flux[index_norm])
    flux=flux/np.mean(flux[index_norm])
    #error = error / np.mean(flux[index_norm])
    #else: print("redshift is out of the range!")
    print('spec_norm_v1.line101 index_norm = ', index_norm)
    print('spec_norm_v1.line114 norm_f = ', norm_f)
    print('spec_norm_v1.line117 flux = ', flux)	

    return flux, norm_f

def spec_rebin(oldwave, oldflux, newwave):
    #print 'size:', oldwave.size, newwave.size
    print('oldwave = ', oldwave)
    print('oldflux = ', oldflux)
    print('newwave = ', newwave)
    print('oldwave_shape = ', oldwave.shape)
    print('oldflux_shape = ', oldflux.shape)
    print('newwave_shape = ', newwave.shape)
    inbetween=(oldwave[-1] >= newwave) & (oldwave[0] <= newwave)
    print('inbetween = ', inbetween)
    print('inbetween_shape = ', inbetween.shape)
    #inbetween=(np.max(oldwave) >= newwave) & (np.min(oldwave) <= newwave)
    new_flux=np.zeros(newwave.size)
    print('new_flux = ', new_flux)
    f = interpolate.interp1d(oldwave, oldflux)
    print('f = ', f)
    new_flux[inbetween]=f(newwave[inbetween])
    print('new_flux = ', new_flux)

    return new_flux



def dust_abs(wave, flux, zabs,feature):
    if feature=='bump' :
        bump_arr=np.arange(20)*0.04+0.5
        index_bump=np.random.randint(bump_arr.size, size=1)
        bump_simu=bump_arr[index_bump]
    if feature=='non':
        bump_simu=0.0
    #inseet bump in the sepctra
    para=np.zeros(5)

    para[3]=4.58
    gamma_simu=0.89
    para[4]=0.89

    para[2]=bump_simu*gamma_simu*2/3.14


    lx=0.05+np.arange(10000)*0.002
    curve=para[2]*lx**2/(lx**2*para[4]**2+(lx**2-para[3]**2)**2)+para[0]+para[1]*lx
    extin=10.0**(-0.4*curve)
    wave1=(10000.0/lx)*(1.0+zabs)
    f_new = interpolate.interp1d(wave1, extin)
    nextin=f_new(wave)
    newflux=flux*nextin

    return newflux



def Iterstat(Inputarr, sigrej, maxiter):
    mask_iter=np.zeros(Inputarr.size)+1
    ngood=Inputarr.size
    Fmean=np.sum(1.*Inputarr*mask_iter)/ngood
    Fsig=np.sqrt(np.sum((1.*Inputarr-Fmean)**2*mask_iter)/(ngood-1))
    print('Interstat_line_177, mask_iter\n', mask_iter)
    print('Interstat_line_178, ngood\n', ngood)
    print('Interstat_line_179, Fmean\n', Fmean)
    print('Interstat_line_180, Fsig\n', Fsig)
    nlast = -1
    Iter  =  0
    while (Iter < maxiter) and (ngood >= 2):
        Lolim=Fmean - sigrej*Fsig
        Hilim=Fmean + sigrej*Fsig
        nlast=ngood
        print('Interstat_line_188, Lolim\n', Lolim)
        print('Interstat_line_189, Hilim\n', Hilim)
        print('Interstat_line_190, ngood\n', ngood)

        mask_iter[Inputarr < Lolim]= 0
        print('Interstat_line_195, mask_iter\n', mask_iter)
        mask_iter[Inputarr > Hilim]= 0
        print('Interstat_line_196, mask_iter\n', mask_iter)
        ngood=np.sum(mask_iter)
        print('Interstat_line_199, ngood\n', ngood)
        Fmean=np.sum(1.*Inputarr*mask_iter)/ngood
        print('Interstat_line_201, Fmean\n', Fmean)
        Fsig=np.sqrt(np.sum((1.*Inputarr-Fmean)**2*mask_iter)/(ngood-1))
        print('Interstat_line_203, Fsig\n', Fsig)
        Iter = Iter+1
        print('Interstat_line_203, Iter\n', Iter)
		
    return Fmean, Fsig

'''
def gaussian_profile(x, mu, sig):
    div = x - np.roll(x, +1)
    index_profile=(mu+6*sig >= x) & (mu-6*sig <= x)
    profile = -1.0/np.sqrt(2*np.pi*sig**2)*np.exp(-(x-mu)**2/(2*sig**2))
    scale=np.abs(np.sum(profile[index_profile]*div[index_profile]))
    #print scale
    profile = profile/scale

    return profile



def absorption_line_model(wave, flux, zqso, zabs, line_list, ew_list, error):
    #calcualate the width of absorption profile
    line_wave=line_list*(1+zabs)
    #resolution=2000
    sig_list=ew_list[-1]*(1+zabs)/2.355
    #print sig_list#line_wave/(resolution*2.355)


    line_model=np.zeros((wave.size, line_list.size))
    for i in np.arange(line_list.size):
#cacluate the coressponding error for lines
        index_error = (wave >= line_wave[i]-3*sig_list) & (wave <= line_wave[i]+3*sig_list)
        snr_line=np.mean(flux[index_error]/error[index_error])
        #print snr_line
        #print 2.355*sig_list/(snr_line*(1+zabs))

        line_model[:,i]=gaussian_profile(wave, line_wave[i], sig_list)*ew_list[i]*(1+zabs)
        #try reverse it here:
    abs_model=np.sum(line_model, axis=1)+1
    flux_abs=flux*abs_model

    return flux_abs, abs_model

'''
