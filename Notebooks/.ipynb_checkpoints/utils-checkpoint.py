#Import necesary modules
#Use shorter names (np, pd, plt) instead of full (numpy, pandas, matplotlib.pylot) for convenience
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import h5py
import xarray as xr
import numpy as np
import pdb
import numpy.ma as ma

def getSnowandConverttoThickness(dF, snowDepthVar='snowDepth', 
                                 snowDensityVar='snowDensity',
                                 outVar='iceThickness'):
    """ Grid using nearest neighbour the NESOSIM snow depths to the 
    high-res ICESat-1 freeboard locations
    """
    
    # Convert freeboard to thickness
    # Need to copy arrays or it will overwrite the pandas column!
    freeboardT=np.copy(dF['freeboard'].values)
    snowDepthT=np.copy(dF[snowDepthVar].values)
    snowDensityT=np.copy(dF[snowDensityVar].values)
    ice_thickness = freeboard_to_thickness(freeboardT, snowDepthT, snowDensityT)
    #print(ice_thickness)
    dF[outVar] = pd.Series(np.array(ice_thickness), index=dF.index)
   
    return dF

def freeboard_to_thickness(freeboardT, snow_depthT, snow_densityT):
    """
    Hydrostatic equilibrium equation to calculate sea ice thickness 
    from freeboard and snow depth/density data

    Args:
        freeboardT (var): ice freeboard
        snow_depthT (var): snow depth
        snow_densityT (var): final snow density

    Returns:
        ice_thicknessT (var): ice thickness dereived using hydrostatic equilibrium

    """

    # Define density values
    rho_w=1024.
    rho_i=925.
    #rho_s=300.

    # set snow to freeboard where it's bigger than freeboard.
    snow_depthT[snow_depthT>freeboardT]=freeboardT[snow_depthT>freeboardT]

    ice_thicknessT = (rho_w/(rho_w-rho_i))*freeboardT - ((rho_w-snow_densityT)/(rho_w-rho_i))*snow_depthT

    return ice_thicknessT

def getWarrenData(dF, outSnowVar, outDensityVar='None'):
    """
    Assign Warren1999 snow dept/density climatology to dataframe

    Added 

    Args:
        dF (data frame): Pandas dataframe
        outSnowVar (string): name of Warren snow depth variable
        outDensityVar (string): name of Warren snow density variable
        

    Returns:
        dF (data frame): Pandas dataframe updated to include colocated Warren snow depth and density
        
    """
    
    # Generate empty lists
    snowDepthW99s=ma.masked_all(np.size(dF['freeboard'].values))
    if (outDensityVar!='None'):
        snowDensityW99s=ma.masked_all(np.size(dF['freeboard'].values))

    # Loop over all freeboard values (rows)
    for x in range(np.size(dF['freeboard'].values)):
        #print(x, dF['lon'].iloc[x], dF['lat'].iloc[x], dF['month'].iloc[x]-1)
        # SUbtract 1 from month as warren index in fucntion starts at 0
        snowDepthDayW99T, snowDensityW99T=WarrenClimatology(dF['lon'].iloc[x], dF['lat'].iloc[x], dF['month'].iloc[x]-1)
        

        # Append values to list
        snowDepthW99s[x]=snowDepthDayW99T
        if (outDensityVar!='None'):
            snowDensityW99s[x]=snowDensityW99T

    # Assign list to dataframe as a series
    dF[outSnowVar] = pd.Series(snowDepthW99s, index=dF.index)
    if (outDensityVar!='None'):
        dF[outDensityVar] = pd.Series(snowDensityW99s, index=dF.index)
    

    return dF

def WarrenClimatology(lonT, latT, monthT):
    """
    Get Warren1999 snow depth climatology

    Args:
        lonT (var): longitude
        latT (var): latitude
        monthT (var): month with the index starting at 0
        
    Returns:
        Hs (var): Snow depth (m)
        rho_s (var): Snow density (kg/m^3)
        
    """

    H_0 = [28.01, 30.28, 33.89, 36.8, 36.93, 36.59, 11.02, 4.64, 15.81, 22.66, 25.57, 26.67]
    a = [.127, .1056, .5486, .4046, .0214, .7021, .3008, .31, .2119, .3594, .1496, -0.1876]
    b = [-1.1833, -0.5908, -0.1996, -0.4005, -1.1795, -1.4819, -1.2591, -0.635, -1.0292, -1.3483, -1.4643, -1.4229]
    c = [-0.1164, -0.0263, 0.0280, 0.0256, -0.1076, -0.1195, -0.0811, -0.0655, -0.0868, -0.1063, -0.1409, -0.1413]
    d = [-0.0051, -0.0049, 0.0216, 0.0024, -0.0244, -0.0009, -0.0043, 0.0059, -0.0177, 0.0051, -0.0079, -0.0316]
    e = [0.0243, 0.0044, -0.0176, -0.0641, -0.0142, -0.0603, -0.0959, -0.0005, -0.0723, -0.0577, -0.0258, -0.0029]

    # Convert lat and lon into degrees of arc, +x axis along 0 degrees longitude and +y axis along 90E longitude
    x = (90.0 - latT)*np.cos(lonT * np.pi/180.0)  
    y = (90.0 - latT)*np.sin(lonT*np.pi/180.0) 

    Hs = H_0[monthT] + a[monthT]*x + b[monthT]*y + c[monthT]*x*y + (d[monthT]*x*x) + (e[monthT]*y*y)
    

    # Now get SWE, although this is not returned by the function

    H_0swe = [8.37, 9.43,10.74,11.67,11.8,12.48,4.01,1.08,3.84,6.24,7.54,8.0]
    aswe = [-0.027,0.0058,0.1618,0.0841,-0.0043,0.2084,0.097,0.0712,0.0393,0.1158,0.0567,-0.054]
    bswe = [-0.34,-0.1309,0.0276,-0.1328,-0.4284,-0.5739,-0.493,-0.145,-0.2107,-0.2803,-0.3201,-0.365]
    cswe = [-0.0319,0.0017,0.0213,0.0081,-0.038,-0.0468,-0.0333,-0.0155,-0.0182,-0.0215,-0.0284,-0.0362]
    dswe = [-0.0056,-0.0021,0.0076,-0.0003,-0.0071,-0.0023,-0.0026,0.0014,-0.0053,0.0015,-0.0032,-0.0112]
    eswe = [-0.0005,-0.0072,-0.0125,-0.0301,-0.0063,-0.0253,-0.0343,0,-0.019,-0.0176,-0.0129,-0.0035]


    swe = H_0swe[monthT] + aswe[monthT]*x + bswe[monthT]*y + cswe[monthT]*x*y + dswe[monthT]*x*x + eswe[monthT]*y*y

    # Density in kg/m^3
    rho_s = 1000.*(swe/Hs)  
    #print(ma.mean(rho_s))

    # Could mask out bad regions (i.e. land) here if desired.
    # Hsw[where(region_maskG<9.6)]=np.nan
    # Hsw[where(region_maskG==14)]=np.nan
    # Hsw[where(region_maskG>15.5)]=np.nan

    # Could mask out bad regions (i.e. land) here if desired.
    #rho_s[where(region_maskG<9.6)]=np.nan
    #rho_s[where(region_maskG==14)]=np.nan
    #rho_s[where(region_maskG>15.5)]=np.nan

    # Convert snow depth to meters
    Hs=Hs/100.

    return Hs, rho_s

def get_psnlatslons(data_path, res=25):
    # Get NSIDC polar stereographic grid data
    if (res==25):
        # 25 km grid
        mask_latf = open(data_path+'/psn25lats_v3.dat', 'rb')
        mask_lonf = open(data_path+'/psn25lons_v3.dat', 'rb')
        lats_mask = reshape(fromfile(file=mask_latf, dtype='<i4')/100000., [448, 304])
        lons_mask = reshape(fromfile(file=mask_lonf, dtype='<i4')/100000., [448, 304])
    elif (res==12):
        # 12.5 km grid
        mask_latf = open(data_path+'/psn12lats_v3.dat', 'rb')
        mask_lonf = open(data_path+'/psn12lons_v3.dat', 'rb')
        lats_mask = reshape(fromfile(file=mask_latf, dtype='<i4')/100000., [896, 608])
        lons_mask = reshape(fromfile(file=mask_lonf, dtype='<i4')/100000., [896, 608])
    elif (res==6):
        # 12.5 km grid
        mask_latf = open(data_path+'/psn06lats_v3.dat', 'rb')
        mask_lonf = open(data_path+'/psn06lons_v3.dat', 'rb')
        lats_mask = reshape(fromfile(file=mask_latf, dtype='<i4')/100000., [1792, 1216])
        lons_mask = reshape(fromfile(file=mask_lonf, dtype='<i4')/100000., [1792, 1216])

    return lats_mask, lons_mask


def assignRegionMask(dF, mapProj, ancDataPath='../Data/'):
    """
    Grab the NSIDC region mask and add to dataframe as a new column

    # 1   non-region oceans
    # 2   Sea of Okhotsk and Japan
    # 3   Bering Sea
    # 4   Hudson Bay
    # 5   Gulf of St. Lawrence
    # 6   Baffin Bay/Davis Strait/Labrador Sea
    # 7   Greenland Sea
    # 8   Barents Seas
    # 9   Kara
    # 10   Laptev
    # 11   E. Siberian
    # 12   Chukchi
    # 13   Beaufort
    # 14   Canadian Archipelago
    # 15   Arctic Ocean
    # 20   Land
    # 21   Coast

    Args:
        dF (data frame): original data frame
        mapProj (basemap instance): basemap map projection
          
    Returns:
        dF (data frame): data frame including ice type column (1 = multiyear ice, 0 = everything else)

    """
    region_mask, xptsI, yptsI = get_region_mask_sect(ancDataPath, mapProj, xypts_return=1)

    xptsI=xptsI.flatten()
    yptsI=yptsI.flatten()
    region_mask=region_mask.flatten()

    #iceTypeGs=[]
    regionFlags=ma.masked_all((size(dF['freeboard'].values)))
    for x in range(size(dF['freeboard'].values)):
        # Find nearest ice type
        dist=sqrt((xptsI-dF['xpts'].iloc[x])**2+(yptsI-dF['ypts'].iloc[x])**2)
        index_min = np.argmin(dist)
        regionFlags[x]=int(region_mask[index_min])
        
        # This is what I sometimes do but it appears slower in this case..
        # I checked and they gave the same answers
        # iceTypeG2 = griddata((xpts_type, ypts_type), ice_typeT2, (dF['xpts'].iloc[x], dF['ypts'].iloc[x]), method='nearest') 
        # print(iceTypeG)
        # iceTypeGs.append(iceTypeG)

    dF['region_flag'] = pd.Series(regionFlags, index=dF.index)

    return dF

def get_region_mask_sect(datapath, mplot, xypts_return=0):
    datatype='uint8'
    file_mask = datapath+'/sect_fixed_n.msk'
    # 1   non-region oceans
    # 2   Sea of Okhotsk and Japan
    # 3   Bering Sea
    # 4   Hudson Bay
    # 5   Gulf of St. Lawrence
    # 6   Baffin Bay/Davis Strait/Labrador Sea
    # 7   Greenland Sea
    # 8   Barents Seas
    # 9   Kara
    # 10   Laptev
    # 11   E. Siberian
    # 12   Chukchi
    # 13   Beaufort
    # 14   Canadian Archipelago
    # 15   Arctic Ocean
    # 20   Land
    # 21   Coast
    fd = open(file_mask, 'rb')
    region_mask = fromfile(file=fd, dtype=datatype)
    region_mask = reshape(region_mask, [448, 304])

    #xpts, ypts = mplot(lons_mask, lats_mask)
    if (xypts_return==1):
        mask_latf = open(datapath+'/psn25lats_v3.dat', 'rb')
        mask_lonf = open(datapath+'/psn25lons_v3.dat', 'rb')
        lats_mask = reshape(fromfile(file=mask_latf, dtype='<i4')/100000., [448, 304])
        lons_mask = reshape(fromfile(file=mask_lonf, dtype='<i4')/100000., [448, 304])

        xpts, ypts = mplot(lons_mask, lats_mask)

        return region_mask, xpts, ypts
    else:
        return region_mask
    
    


"""
convert_GPS_time.py (10/2017)
Return the calendar date and time for given GPS time.
Based on Tiffany Summerscales's PHP conversion algorithm
	https://www.andrews.edu/~tzs/timeconv/timealgorithm.html

INPUTS:
	GPS_Time: GPS time (standard = seconds since January 6, 1980 at 00:00)

OUTPUTS:
	month: Number of the desired month (1 = January, ..., 12 = December).
	day: Number of day of the month.
	year: Number of the desired year.
	hour: hour of the day
	minute: minute of the hour
	second: second (and fractions of a second) of the minute.

OPTIONS:
	OFFSET: number of seconds to offset each GPS time

PYTHON DEPENDENCIES:
	numpy: Scientific Computing Tools For Python (http://www.numpy.org)

PROGRAM DEPENDENCIES:
	convert_julian.py: convert Julian dates into calendar dates

UPDATE HISTORY:
	Updated 10/2017: added leap second from midnight 2016-12-31
	Written 04/2016
"""


#-- PURPOSE: Define GPS leap seconds
def get_leaps():
	leaps = [46828800, 78364801, 109900802, 173059203, 252028804, 315187205,
		346723206, 393984007, 425520008, 457056009, 504489610, 551750411,
		599184012, 820108813, 914803214, 1025136015, 1119744016, 1167264017]
	return leaps

#-- PURPOSE: Test to see if any GPS seconds are leap seconds
def is_leap(GPS_Time):
	leaps = get_leaps()
	Flag = np.zeros_like(GPS_Time, dtype=np.bool)
	for leap in leaps:
		count = np.count_nonzero(np.floor(GPS_Time) == leap)
		if (count > 0):
			indices, = np.nonzero(np.floor(GPS_Time) == leap)
			Flag[indices] = True
	return Flag

#-- PURPOSE: Count number of leap seconds that have passed for each GPS time
def count_leaps(GPS_Time):
	leaps = get_leaps()
	#-- number of leap seconds prior to GPS_Time
	n_leaps = np.zeros_like(GPS_Time, dtype=np.uint)
	for i,leap in enumerate(leaps):
		count = np.count_nonzero(GPS_Time >= leap)
		if (count > 0):
			indices, = np.nonzero(GPS_Time >= leap)
			# print(indices)
			# pdb.set_trace()
			n_leaps[indices] += 1
	return n_leaps

#-- PURPOSE: Convert UNIX Time to GPS Time
def convert_UNIX_to_GPS(UNIX_Time):
	#-- calculate offsets for UNIX times that occur during leap seconds
	offset = np.zeros_like(UNIX_Time)
	count = np.count_nonzero((UNIX_Time % 1) != 0)
	if (count > 0):
		indices, = np.nonzero((UNIX_Time % 1) != 0)
		UNIX_Time[indices] -= 0.5
		offset[indices] = 1.0
	#-- convert UNIX_Time to GPS without taking into account leap seconds
	#-- (UNIX epoch: Jan 1, 1970 00:00:00, GPS epoch: Jan 6, 1980 00:00:00)
	GPS_Time = UNIX_Time - 315964800
	leaps = get_leaps()
	#-- calculate number of leap seconds prior to GPS_Time
	n_leaps = np.zeros_like(GPS_Time, dtype=np.uint)
	for i,leap in enumerate(leaps):
		count = np.count_nonzero(GPS_Time >= (leap - i))
		if (count > 0):
			indices, = np.nonzero(GPS_Time >= (leap - i))
			n_leaps[indices] += 1
	#-- take into account leap seconds and offsets
	GPS_Time += n_leaps + offset
	return GPS_Time

#-- PURPOSE: Convert GPS Time to UNIX Time
def convert_GPS_to_UNIX(GPS_Time):
	#-- convert GPS_Time to UNIX without taking into account leap seconds
	#-- (UNIX epoch: Jan 1, 1970 00:00:00, GPS epoch: Jan 6, 1980 00:00:00)
	UNIX_Time = GPS_Time + 315964800
	#-- number of leap seconds prior to GPS_Time
	n_leaps = count_leaps(GPS_Time)
	UNIX_Time -= n_leaps
	#-- check if GPS Time is leap second
	Flag = is_leap(GPS_Time)
	if Flag.any():
		#-- for leap seconds: add a half second offset
		indices, = np.nonzero(Flag)
		UNIX_Time[indices] += 0.5
	return UNIX_Time

#-- PURPOSE: convert from GPS time to calendar dates
def convert_GPS_time(GPS_Time, OFFSET=0.0):
	#-- convert from standard GPS time to UNIX time accounting for leap seconds
	#-- and adding the specified offset to GPS_Time
	UNIX_Time = convert_GPS_to_UNIX(np.array(GPS_Time) + OFFSET)
	#-- calculate Julian date from UNIX time and convert into calendar dates
	#-- UNIX time: seconds from 1970-01-01 00:00:00 UTC
	julian_date = (UNIX_Time/86400.0) + 2440587.500000
	cal_date = convert_julian(julian_date)
	#-- include UNIX times in output
	cal_date['UNIX'] = UNIX_Time
	#-- return the calendar dates and UNIX time
	return cal_date
def convert_julian(JD, ASTYPE=None, FORMAT='dict'):
	#-- convert to array if only a single value was imported
	if (np.ndim(JD) == 0):
		JD = np.array([JD])
		SINGLE_VALUE = True
	else:
		SINGLE_VALUE = False

	JDO = np.floor(JD + 0.5)
	C = np.zeros_like(JD)
	#-- calculate C for dates before and after the switch to Gregorian
	IGREG = 2299161.0
	ind1, = np.nonzero(JDO < IGREG)
	C[ind1] = JDO[ind1] + 1524.0
	ind2, = np.nonzero(JDO >= IGREG)
	B = np.floor((JDO[ind2] - 1867216.25)/36524.25)
	C[ind2] = JDO[ind2] + B - np.floor(B/4.0) + 1525.0
	#-- calculate coefficients for date conversion
	D = np.floor((C - 122.1)/365.25)
	E = np.floor((365.0 * D) + np.floor(D/4.0))
	F = np.floor((C - E)/30.6001)
	#-- calculate day, month, year and hour
	DAY = np.floor(C - E + 0.5) - np.floor(30.6001*F)
	MONTH = F - 1.0 - 12.0*np.floor(F/14.0)
	YEAR = D - 4715.0 - np.floor((7.0+MONTH)/10.0)
	HOUR = np.floor(24.0*(JD + 0.5 - JDO))
	#-- calculate minute and second
	G = (JD + 0.5 - JDO) - HOUR/24.0
	MINUTE = np.floor(G*1440.0)
	SECOND = (G - MINUTE/1440.0) * 86400.0

	#-- convert all variables to output type (from float)
	if ASTYPE is not None:
		YEAR = YEAR.astype(ASTYPE)
		MONTH = MONTH.astype(ASTYPE)
		DAY = DAY.astype(ASTYPE)
		HOUR = HOUR.astype(ASTYPE)
		MINUTE = MINUTE.astype(ASTYPE)
		SECOND = SECOND.astype(ASTYPE)

	#-- if only a single value was imported initially: remove singleton dims
	if SINGLE_VALUE:
		YEAR = YEAR.item(0)
		MONTH = MONTH.item(0)
		DAY = DAY.item(0)
		HOUR = HOUR.item(0)
		MINUTE = MINUTE.item(0)
		SECOND = SECOND.item(0)

	#-- return date variables in output format (default python dictionary)
	if (FORMAT == 'dict'):
		return dict(year=YEAR, month=MONTH, day=DAY,
			hour=HOUR, minute=MINUTE, second=SECOND)
	elif (FORMAT == 'tuple'):
		return (YEAR, MONTH, DAY, HOUR, MINUTE, SECOND)
	elif (FORMAT == 'zip'):
		return zip(YEAR, MONTH, DAY, HOUR, MINUTE, SECOND)
