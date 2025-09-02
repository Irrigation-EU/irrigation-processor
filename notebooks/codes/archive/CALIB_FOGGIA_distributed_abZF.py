#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import datetime as dt
import pandas as pd
from scipy.optimize import minimize
import h5py


# In[2]:


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


# In[3]:


AGG = 7       
T = 2


print ('RF VALUE IS CALIBRATED')



#CALIBRATION
def ts_smet4irr(sm, et, a, b, z, RF, thr=None): 
    # sm - soil moisture
    # et - evotranspiration
    # a/b - drainage parameter
    # z - soil water capacity
    # f/RF - Adjusting factor
    p_sim = z * (sm[1:] - sm[:-1]) + ((a * sm[1:]**b + a * sm[:-1]**b) / 2.) +  ((RF * sm[1:] * et[1:] + RF * sm[:-1] * et[:-1]) / 2.)

    p_sim[abs(np.diff(sm))<=0.001]=0.0
    p_sim[p_sim < 1.0]=0.0   # 2mm/day for NN=4 -> 0.5

    return np.clip(p_sim, 0, thr)

def calib_smet4irr(sm,  p_obs, et, NN,   x0=None, bounds=None, options=None, method='TNC'): 


    if x0 is None:
        x0 = np.array([20., 5., 80,1.])

    if bounds is None:
        bounds = ((0,200), (0.01, 50), (1, 800), (0.1,1.4))

    if options is None:
        options = {'ftol': 1e-8, 'maxiter': 4000, 'disp': False}

#    if options is None:
#        options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}


    result = minimize(cost_fun, x0, args=(sm,  p_obs, et, NN),method=method, bounds=bounds, options=options)

    a, b, z,RF = result.x

    return a, b, z, RF


def cost_fun(x0, sm,  p_obs, et, NN):

    # The following args are 1D time-series
    p_sim = ts_smet4irr(sm, et, x0[0], x0[1], x0[2],x0[3])
    p_sim[np.isnan(p_obs)] = np.nan
    p_sim1 = np.add.reduceat(p_sim[np.isfinite(p_sim)], np.arange(0, len(p_sim[np.isfinite(p_sim)]), NN))
    p_obs1 = np.add.reduceat(p_obs[np.isfinite(p_sim)], np.arange(0, len(p_obs[np.isfinite(p_sim)]), NN))
    rmsd = np.nanmean((p_obs1 - p_sim1)**2)**0.5

    return rmsd

def soil_water_index(sm, jd, t=2.):

    n = sm.size

    swi = np.zeros(n)
    swi[np.isnan(sm)] = np.nan

    k = np.ones(n)
    k[np.isnan(sm)] = np.nan

    idx = np.arange(n)[~np.isnan(sm)]
    n_not_nan = idx.size

    for i in np.arange(1, n_not_nan):
        dt = jd[idx[i]] - jd[idx[i-1]]
        k[idx[i]] = k[idx[i-1]] / (k[idx[i-1]] + np.exp(-dt/t))
        swi[idx[i]] = swi[idx[i-1]] + k[idx[i]] * (sm[idx[i]] - swi[idx[i-1]])

    return swi, k
    
def swicomp_nan(in_data, in_jd, ctime:int=2):
    # Soil moisture computation
    filtered = np.empty(len(in_data))
    gain = 1
    filtered.fill(np.nan)

    ID = np.where(~np.isnan(in_data))
    D = in_jd[ID]
    SWI = in_data[ID]
    tdiff = np.diff(D)

    # find the first non nan value in the time series

    for i in range(2, SWI.size):
        gain = gain / (gain + np.exp(- tdiff[i - 1] / ctime))
        SWI[i] = SWI[i - 1] + gain * (SWI[i] - SWI[i-1])

    filtered[ID] = SWI

    return filtered


# # INPUT DATA

# In[6]:

datasm = h5py.File('C:/Projects/IrrigationEU/Puglia_exp/ELAB/S1_18_24_REGRIDDED.hdf5','r')
sm = np.array(datasm['sm'][:,:,:])
datasm.close()

datapet = Dataset('C:/Projects/IrrigationEU/Puglia_exp/ELAB/ERA5_Land_pet_18_24_REGRIDDED.hdf5','r')
petdaily = np.array(datapet['pet'][:,:,:]) 
datapet.close()

datatp = Dataset('C:/Projects/IrrigationEU/Puglia_exp/ELAB/ERA5_Land_rain_18_24_REGRIDDED.hdf5','r')
tp = np.array(datatp['tp'][:,:,:])
datatp.close()


print (sm.shape)
print (petdaily.shape)
print (tp.shape)

print (np.nanmax(sm), np.nanmin(sm))



datex = [dt.datetime(2018,1,1) + dt.timedelta(days = i) for i in range(sm.shape[0])]
print (datex[0])
print (datex[-1])
#print datex




for i in range(petdaily.shape[1]):
    for j in range(petdaily.shape[2]):
        y = petdaily[:,i,j]
        B = np.nanmean(y,axis = 0)
        if np.isfinite(B) == True:
            nans, x= nan_helper(y)
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])



JDAYSS = pd.Series(datex,datex)
JDAYS = JDAYSS.index.to_julian_date().to_numpy()


# In[10]:


for i in range(sm.shape[1]):
    for j in range(sm.shape[2]):
        y = sm[:,i,j]
        A = np.nanmean(y,axis = 0)
        if np.isfinite(A) == True:
            nans, x= nan_helper(y)
            y[nans] = np.interp(x(nans), x(~nans), y[~nans])


# In[ ]:

print ('CALCULATING SWI')
SWI = np.empty((sm.shape))
for i in range(sm.shape[1]):
    for j in range(sm.shape[2]):
        SWI[:,i,j] = swicomp_nan(sm[:,i,j], JDAYS,T)
                              
plt.imshow(np.nanmean(SWI,axis = 0))
plt.colorbar()
plt.show()
plt.close()

# MASKING p obs = 0 to nan during summer

# In[ ]:

print ('MASKING IRRIGATION SEASON DAYS WITH NO RAINFALL (less than 1 mm)')
rain_flat = np.empty((tp.shape[0], tp.shape[1]*tp.shape[2]))
for i in range(tp.shape[0]):
    rain_flat[i,:] = tp[i,:,:].reshape(tp.shape[1]*tp.shape[2])
    
RAIN_dataframe = pd.DataFrame(rain_flat[:-1,:], index = datex[:-1])

mask1 = (RAIN_dataframe.index.month > 4) 
mask2 = (RAIN_dataframe.index.month < 10)
mask3 = (RAIN_dataframe[mask1 & mask2]) < 1.0

RAIN_dataframe[mask3] = np.nan


rain = np.array(RAIN_dataframe)


rainfall = np.empty((rain.shape[0], tp.shape[1],tp.shape[2]))
for i in range(rain.shape[0]):
    rainfall[i,:,:] = rain[i,:].reshape(tp.shape[1],tp.shape[2])
    



# In[ ]:


SWI_masked_flat = np.empty((SWI.shape[0], SWI.shape[1]*SWI.shape[2]))
for i in range(SWI.shape[0]):
    SWI_masked_flat[i,:] = SWI[i,:,:].reshape(SWI.shape[1]*SWI.shape[2])


# In[ ]:


rainfall_masked_flat = np.empty((rainfall.shape[0], rainfall.shape[1]*rainfall.shape[2]))
for i in range(rainfall.shape[0]):
    rainfall_masked_flat[i,:] = rainfall[i,:,:].reshape(rainfall.shape[1]*rainfall.shape[2])


# In[ ]:


petdaily_masked_flat = np.empty((petdaily.shape[0], petdaily.shape[1]*petdaily.shape[2]))
for i in range(petdaily.shape[0]):
    petdaily_masked_flat[i,:] = petdaily[i,:,:].reshape(petdaily.shape[1]*petdaily.shape[2])


# In[ ]:

print ('FLATTEN ARRAYS')
print (SWI_masked_flat.shape)
print (rainfall_masked_flat.shape)
print (petdaily_masked_flat.shape)


print ('CALIBRATION OF a, b,  Z, and RF')

a = np.empty((sm.shape[1]*sm.shape[2]))
b = np.empty((sm.shape[1]*sm.shape[2]))
z = np.empty((sm.shape[1]*sm.shape[2]))
f = np.empty((sm.shape[1]*sm.shape[2]))

for i in range(sm.shape[1]*sm.shape[2]):

    if np.isnan(np.nanmean(SWI_masked_flat[:,i])):
        
        a[i] = np.nan
        b[i] = np.nan
        z[i] = np.nan
        f[i] = np.nan
    else:
        print ('VALID POINT', i)
        H = calib_smet4irr(SWI_masked_flat[:,i],rainfall_masked_flat[:,i],petdaily_masked_flat[:,i],AGG ) 
        a[i] = H[0]
        b[i] = H[1]
        z[i] = H[2]
        f[i] = H[3]
        

plt.imshow(f.reshape(sm.shape[1],sm.shape[2]))
plt.colorbar()
plt.show()

np.savetxt(r'a_FOGGIA_distributed_20182024.txt',a.reshape(sm.shape[1],sm.shape[2]))
np.savetxt(r'b_FOGGIA_distributed_20182024',b.reshape(sm.shape[1],sm.shape[2]))
np.savetxt(r'Z_FOGGIA_distributed_20182024',z.reshape(sm.shape[1],sm.shape[2]))
np.savetxt(r'F_FOGGIA_distributed_20182024',f.reshape(sm.shape[1],sm.shape[2]))