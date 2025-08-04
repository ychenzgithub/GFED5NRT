import os
""" GFED5e NRT
   ----------------

    This is a stand-alone python code to run the GFED5 extension NRT version (GFED5eNRT) burned area (BA) and emissions (EM).
   
    In this code, we derived the extension version of daily NRT GFED5 burned area and emissions (at 0.25 deg resolution) from the VIIRS 375m active fire data. The VIIRS active fire counts are recorded and categorized to different burning types based on land cover types. The GFED5eNRT burned area and emissions are calculated based on 2-step scaling approaches. First, pre-derived effective fire area (EFA) scalars were used to convert the daily VIIRS active fire number (VAF) to daily burned area at 0.25 degree resolution (GFED5eNRT BA). Then, pre-calculated fuel consumption (FC) scalars were combined with GFED5eNRT BA to derive daily burned area at 0.25 degree resolution (GFED5eNRT EM). 

    This code is able to automatically
        * Search for the days needed to update the GFED5e NRT data
        * For each day,
            1) Read and update VIIRS 375m active fire data (either NRT data from LANCE or standard data from LAADS DAAC)
            2) Record VAF to 500m resolution (including scan angle correction)
            3) Record VAF over burning types for each 0.25 deg grid cell (including deforestation correction)
            4) Use EFA scalar to convert VAF to BA
            5) Use FC scalar to convert BA to EM

    The current code can be run on redwood.ess.uci.edu. 
    |__Code: The current python script code
    |__Input: The needed input data
            - Earthdata_token.txt: NASA Earthdata token (may need regular update)

            - GFED51VFA_regtp.csv: scalar to convert VAF to BA
            - GFED51FC_regtp.csv: scalar to convert BA to EM
            - VIIRSsplwgt_2019-2021.csv: weights for scan angle correction 
            - GFED51_EF.csv: table of emission factors

            - MaskDefonly: 500m Deforestation only mask (not overlapped with MaskPeat)
            - Sample.MCD64A1.hdf: sample 500m MODIS BA data (used for data structure only)

            - fVAF_Defo_2013-2022.nc: climatological mean deforestation fraction for active fires
            - CFmean_2014-2023.nc: climatological mean cloud fraction
            - fraction_forest_BA.txt: 0.25 deg forest fraction averaged over 2001-2003
            - GFEDregions_025d.tif: 0.25 deg GFED region mask
            - mjLCT_MCD12C1_0.25x0.25.hdf: 0.25 deg major land cover type data
            - MCD12Q1_LCd_025d_2022.tif: 0.25 deg GFED5 land cover data for 2022

            - VNP14IMG: VIIRS monthly standard active fire data (from fuoco)
            - VNP14IMGDL: VIIRS daily standard active fire data (from LAADS DAAC)
            - VNP14IMG_NRT: VIIRS daily NRT active fire data (from LANCE)
            - MOD500mLatLon: MODIS 500m latitude and longitude data
            - MaskPeat: 500m Peatland mask
            - MaskDef_2022: 500m Deforestation mask for 2022
            - MCD12Q1_LCd_2022: 500m MODIS land cover data for 2022
    |__Intermediate: The intermediate data created during the running of this script
            - VIIRSFC: 500m VAF data
    |__Output: The output data
            - BA: GFED5e NRT BA
            - EM: GFED5e NRT EM
            - VAF: daily VAF at 0.25 degree

    Burning type classification
        * Four burning type classification schemes: Modified MODIS LCT (MOD), GFED5 BA LCT (GBA), GFED5 16-class LCT (G16), GFED5 6-class LCT (G6)
            * EFA - GBA
            * FC - G16 & G6      
            * VFA - GBA
            * BA - G16 (by converting VFA from GBA to G16 in the VFA to BA scaling) & G6
            * EM - G16 & G6

    Running instructions
    ---------------------
    - Set python environment
        The following external python modules are needed for running the code: pyhdf, gdal, xarray, numpy, pandas, pyproj,pynio, scipy.
    - Set running directory
        Modify `dirData` to the running directory.
    - Set NASA Earthdata credential
        Copy your own token to `Earthdata_token.txt` in the `Input` subdirectory.
    - Run the code
"""

# All the data (or links to the data) are located in this directory
# dirData = os.path.expanduser('~/GoogleDrive/My/My.Research/UCI/ProjectData/GFED5/GFED51NRTrun/')
dirData = os.path.expanduser('~/GFED5eNRT/')
dirVNP14IMGML = os.path.join(dirData, 'Input', 'VNP14IMGML')

import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------------

# lat/lon for 0.25 deg grid cell centers
G25_lats = np.linspace(-90+0.25/2,90-0.25/2,720)
G25_lons = np.linspace(-180+0.25/2,180-0.25/2,1440)

# GFED names
GFEDnms = ['OCEA', 'BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS', 'AUST']

# land cover names for GFED5.1 burned area
LCnms = ['Water_W','Boreal_F','Tropic_F','Temper_F','Mosaic_FS', 'Tropic_SH',
        'Temper_SH','Temper_G','Woody_SA', 'Open_SA', 'Tropic_G',
        'Wetland_W', 'Tropic_C', 'Urban_U', 'Temper_C', 'Snow_SI',
         'Barren_B', 'Boreal_FS', 'Tundra_T', 'Boreal_C']         # name
LCnms_full = ['Water','Forest boreal','Forest tropical','Forest temperate','Sparse temperate mosaic','Shrublands tropical','Shrublands temperate','Grasslands temperate','Savanna woody','Savanna open','Grasslands tropical','Wetlands','Croplands tropical','Urban','Croplands temperate','Snow/ice','Barren','Sparse boreal forest','Tundra','Croplands boreal']

# 16-class Land cover names for GFED5.1 emissions
EMLCnms = ['Tundra','Sparse boreal forest','Boreal forest','Temperate grassland','Temperate shrubland','Temperate mosaic','Temperate forest','Tropical grassland','Tropical shrubland','Open savanna','Woody savanna','Tropical forest','Other','Cropland','Peat','Deforestation']

# ----------------------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------------------

def strymd(yr, mo, day):
    """
    Converts a year, month, and day into a zero-padded string in 'YYYY-MM-DD' format.

    Args:
        yr (int): Year.
        mo (int): Month.
        day (int): Day.

    Returns:
        str: Date string in 'YYYY-MM-DD' format.
    """
    strymd = str(yr) + '-' + str(mo).zfill(2) + '-' + str(day).zfill(2)
    return strymd

def strdoy(yr, mo, day):
    """
    Converts a year, month, and day into a zero-padded day-of-year (DOY) string.

    Args:
        yr (int): Year.
        mo (int): Month.
        day (int): Day.

    Returns:
        str: Day of year as a zero-padded 3-character string (e.g., '001' to '366').
    """
    from datetime import datetime
    doy = datetime(yr, mo, day).timetuple().tm_yday
    strdoy = str(doy).zfill(3)
    return strdoy

def mkdir(dirstr, checkonly=False):
    """
    Checks if the specified directory exists and creates it (and parent directories) if it does not.

    Args:
        dirstr (str): Path to the directory.
        checkonly (bool, optional): If True, only checks existence without creating. Default is False.

    Returns:
        None
    """
    import os
    if not os.path.exists(dirstr):
        print(f"Directory '{dirstr}' does not exist. Creating...")
        if checkonly:
            return
        os.makedirs(dirstr, exist_ok=True)
        print(f"Directory '{dirstr}' and its parent directories created successfully.")

def to_netcdf(ds, filename, mode="w", group=None):
    """
    Saves an xarray Dataset to a NetCDF file with compression.

    Args:
        ds (xarray.Dataset): Dataset to save.
        filename (str): Output NetCDF file path.
        mode (str, optional): Write mode ('w' to overwrite, 'a' to append). Default is 'w'.
        group (str, optional): NetCDF group name, if saving to a group. Default is None.

    Returns:
        None
    """
    comp = dict(zlib=True, complevel=5)
    encoding = {
        var: comp for var in ds.data_vars
        if ds[var].dtype.kind in {"f", "i"}  # compress float or int variables
    }
    ds.to_netcdf(filename, encoding=encoding, mode=mode, group=group)

def nowarn():
    """
    Suppresses all Python warnings.

    Returns:
        None
    """
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")

# functions used to read intermediate and output data
def read_BA(yr,mo,day):
    """ read BA dataset from the output directory
    """
    import xarray as xr
    import os
    fnm = dirData+'Intermediate/BA/'+str(yr)+'/BA_'+strymd(yr,mo,day)+'.nc'
    if os.path.exists(fnm):
        ds = xr.open_dataset(fnm)
    else:
        ds = None
    return ds

def read_EM(yr,mo,day):
    """ read EM dataset from the output directory
    """
    import xarray as xr
    ds = xr.open_dataset(dirData+'Intermediate/EM/'+str(yr)+'/EM_'+strymd(yr,mo,day)+'.nc')
    return ds

def read_VNP14IMG_NRT_daily(yr,mo,day):
    import pandas as pd
    from datetime import datetime
    stryr = str(yr)
    doy = datetime(yr,mo,day).timetuple().tm_yday
    strdoy = str(doy).zfill(3)
    fnm = dirData + 'Input/VNP14IMGDL/VNP14IMGDL_'+stryr+strdoy+'.csv'
    df = pd.read_csv(fnm,index_col=0)
    return df

def read_VNP14IMGML_daily(yr,mo,day,ver='C2.02', usecols=["YYYYMMDD", "HHMM", "Lat", "Lon", "Line", "Sample", "FRP", "Confidence", "Type", "DNFlag"]):
    """ Optionally use VIIRS daily data from monthly standard data (VNP14IMGML)
    """    
    # set monthly VNP14IMGML file name
    fnmFC = os.path.join(dirVNP14IMGML,'VNP14IMGML.' + str(yr) + str(mo).zfill(2) + '.' + ver + '.csv')
    
    # read data
    if os.path.exists(fnmFC):
        df = pd.read_csv(
            fnmFC,
            parse_dates=["YYYYMMDD"],
            usecols=usecols,
            skipinitialspace=True,
            low_memory=False
        )
        
         # sometimes the FRP value is '*******' and cause incorrect dtype, need to correct this
        df = df.replace('**********',0)
        df = df.replace('*******',0)
        df['FRP'] = df['FRP'].astype('float')

        # filter to get vegetation fire only
        df = df[df['Type'] == 0]

        # extract daily data
        df = df[df.YYYYMMDD == pd.to_datetime(str(yr)+'-'+str(mo).zfill(2)+'-'+str(day).zfill(2))]

        return df
    else:
        print('No data available for file',fnmFC)
        return None

def read_GFED5eco(yr,mo,day):
    """ read daily GFED5eco dataset from the output directory
    """
    import xarray as xr
    import os

    fnm = dirData+'Output/'+str(yr)+'/GFED5eNRTeco_'+strymd(yr,mo,day)+'.nc'
    if os.path.exists(fnm):
        ds = xr.open_dataset(fnm)
    else:
        ds = None
    return ds

# functions used for processing VIIRS data at 500m MODIS sinusoidal grid cells (Originally in Raster_utility.py and IO_utility.py)
def MODtilegt(H,V,mres=500):
    ''' Get the geotransform information in a MODIS tile (H,V)
    Usage: gt = MODtilegt(18,8)
    
    Parameters
    ----------
    H : int
        h value of MOD tile
    V : int
        v value of MOD tile 
    mres : int, 500|1000
        MODIS resolution (m)
        
    Returns
    -------
    gt : list
        geotransform information 
        # gt[0] /* top left x */
        # gt[1] /* w-e pixel resolution */
        # gt[2] /* rotation, 0 if image is "north up" */
        # gt[3] /* top left y */
        # gt[4] /* rotation, 0 if image is "north up" */
        # gt[5] /* n-s pixel resolution */         
    '''
    R0 = 6371007.181000  # Earth radius in [m]
    limit_left = -20015109.354  # left limit of MODIS grid in [m]
    limit_top = 10007554.677  # top limit of MODIS grid in [m]
    T = R0*np.pi/18.             # m, height/width of MODIS tile        
    ndim = int((500 * 2400) / mres)    
    realres = ((abs(limit_left) * 2) / 36) / ndim    # actual size for each pixel

    # define geotransformation
    gt0 = limit_left + H*T
    gt3 = limit_top - V*T
    gt1 = realres
    gt5 = -realres
    gt = [gt0,gt1,0,gt3,0,gt5]    
    
    return gt

def prj4sinus():
    ''' Return the prj4 string of sinusoidal projection
    '''
    return "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

def sinusproj(xin,yin,inverse=False):
    ''' Project lon, lat to x, y values or vise versa, using the sinusoidal projection
    Usage: x, y = sinusproj(lon,lat);
           lon, lat = sinusproj(x,y,inverse=True)
    
    Parameters
    ----------
    xin : float
        longitude or x value (m)
    yin : float
        latitude or y value (m)
    
    Returns
    -------
    xout : float
        longitude or x value (m)
    yout : float
        latitude or y value (m)
    '''
    
    import pyproj
    
    # set the projection
    prj4str = prj4sinus()
    p_modis_grid = pyproj.Proj(prj4str)
    
    # do the conversion
    xout, yout = p_modis_grid(xin, yin, inverse=inverse)
    
    # alternative method using pyproj Transformer
    #     from pyproj import Transformer
    #     t = Transformer.from_crs(prj4str, "epsg:4326", always_xy=True)
    #     xout, yout = t.transform(xv, yv)
    
    return xout,yout

def FCpoints2arr(dfFCin,gt,xs,ys,strlon='lon',strlat='lat',sumcol=None):
    ''' convert lat/lon pairs to a np array of fire mask, corresponding to a pre-defined image
    
    Parameters
    ----------
    dfFCin : pd DataFrame
        data with 'lon' and 'lat' columns
    gt : list
        the geotransform of the image
    xs : int
        x size of the image
    ys : int
        y size of the image
    buff : bool
        option to extend fire to all 9 pixels around the center (True). if False, only the center pixel is flagged.
    cnt : bool
        if set to True, return number of values per pixel; otherwise, return True/False only
    sumcol : str
        if set, sum the values in this column instead of counting the number of values
    Returns
    -------
    arrFC : np array
        the 2-D fire mask image
    '''
    dfFC = dfFCin.copy()
    
    # derive pixel location (i,j) in the image for each valid active fire pixel   
    dfFC['vx'],dfFC['vy'] = sinusproj(dfFC[strlon].values,dfFC[strlat].values)
    dfFC['i'] = (dfFC['vx']-gt[0])/gt[1]  # column index
    dfFC['i'] = dfFC['i'].astype(int)
    dfFC['j'] = (dfFC['vy']-gt[3])/gt[5]  # row index
    dfFC['j'] = dfFC['j'].astype(int)    
    
    # remove pixels outside of the image boundary
    isval = (dfFC['i']>=0) & (dfFC['i']<xs) & (dfFC['j']>=0) & (dfFC['j']<ys)
    dfFC = dfFC[isval]

    # calculate counts or sum over a column
    if sumcol is None:
        dfgroup = dfFC.groupby(['i','j']).size()
    else:
        dfgroup = dfFC.groupby(['i','j'])[sumcol].sum()

    # loop over each group 
    arrFC = np.zeros([ys,xs],dtype='float')
    for index, value in dfgroup.items():
        ix,iy = index
        arrFC[iy,ix] +=  value
        
    return arrFC

def MODtileij2xy_array(H, V):
    """
    Vectorized version to compute x, y for all i, j in the tile grid.
    """
    import numpy as np
    
    T = 1111950.5197665233
    Xmin = -20015109.355797417
    Ymax = 10007554.677898709
    
    ng = int((500 * 2400) / 500.)  # number of grid cells
    w = T / ng             # size of a grid cell

    # Create i (col) and j (row) grids
    i_grid, j_grid = np.meshgrid(np.arange(2400), np.arange(2400))

    # Compute x and y for the entire grid
    x = (i_grid + 0.5) * w + H * T + Xmin
    y = Ymax - (j_grid + 0.5) * w - V * T

    return x, y

def sinusproj_array(x, y, inverse=False):
    """
    Vectorized sinusoidal projection using pyproj
    """
    import pyproj
    
    prj4str = prj4sinus()
    
    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj(prj4str),  # from
        pyproj.Proj("EPSG:4326"),  # to WGS84 (lon/lat)
        always_xy=True
    ) if inverse else pyproj.Transformer.from_proj(
        pyproj.Proj("EPSG:4326"),  # from WGS84
        pyproj.Proj(prj4str),      # to MODIS sinusoidal
        always_xy=True
    )

    # Apply transform
    lon, lat = transformer.transform(x, y)
    return lon, lat

def getMODlatlon(vh,vv):
    ''' Read lat/lon from a MODIS BA hdf file and derive BA mask
    Usage: arrBAmod = getMODBA(vh,vv)

    Parameters
    ----------
    vh : int
        h value of MODBA
    vv : int
        v value of MODBA

    Returns
    -------
    lon : np array
        2-D lon array
    lat : np array
        2-D lat array
    '''

    # Get x, y coordinates for the MOD tile
    x, y = MODtileij2xy_array(vh, vv)
    
    # Convert x, y to lon, lat using sinusoidal projection
    lon, lat = sinusproj_array(x, y, inverse=True)
    
    return lon,lat

def readMODlatlon(vh,vv):
    ''' Read lat/lon from a MODIS BA hdf file and derive BA mask
    Usage: arrBAmod = getMODBA(vh,vv)

    Parameters
    ----------
    vh : int
        h value of MODBA
    vv : int
        v value of MODBA

    Returns
    -------
    lon : np array
        2-D lon array
    lat : np array
        2-D lat array
    '''
    import xarray as xr

    # get file name
    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)
    f_latlon = dirData+'Input/MOD500mLatLon/latlon500m_'+strhv+'.nc'

    # extract the data
    ds = xr.open_dataset(f_latlon)
    lon = ds.lon.load().values
    lat = ds.lat.load().values

    return lon,lat

def get_tile_paras(vh,vv):
    ''' return parameters of a tile
    '''
    import numpy as np

    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)
    xs, ys = 2400,2400  # number of pixels in each 500m grid cells
    MODgt = MODtilegt(vh,vv)  # read geotransform for the tile
    Tlon,Tlat = getMODlatlon(vh,vv)  # read latitude and longitude of grid cells in the tile
    bb = np.nanmin(Tlat),np.nanmax(Tlat),np.nanmin(Tlon),np.nanmax(Tlon)  # regional box

    return strhv,xs,ys,MODgt,bb

def set_FCtile_ds(arrFC_month):
    ''' Based on the input 2d array of FC data, generate a dataset that is formatted the same as MCD64A1 (using a sample MCD64A1 hdf file as reference)
    '''
    import xarray as xr
    
    # read any 500m MCD64A1 hdf file (4800x4800)
    fnmhdf = dirData+'Input/Sample.MCD64A1.hdf'  
    # dsMOD = xr.open_dataset(fnmhdf,engine='pynio')
    # dsMOD = dsMOD.assign(FC = dsMOD['Burn_Date'])

    dsMOD = xr.open_dataset(fnmhdf,engine='netcdf4')
    dsMOD = dsMOD.assign(FC = dsMOD['Burn Date'])

    # replace values of FC with VAF in each 500m grid cell
    dsMOD['FC'][:] = arrFC_month
    dsFC = xr.Dataset({'FC':dsMOD['FC']})

    return dsFC

# functions used for read 500m ancillary raster data
def tif2arr(fnm):
    ''' Read numpy array from a tif file (first layer)

    Parameters
    ----------
    fnm : str
        The Geotiff file name

    Returns
    -------
    value : np array
        The 2-D array containing the first-layer data of the tif file
    '''
    from osgeo import gdal
    import os
    if os.path.exists(fnm):
        ds = gdal.Open(fnm)
        value = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        return value
    else:
        return None
    
def getMODLC(vh,vv):
    ''' Read MODIS land cover tile data
    Usage: arrLCT = getMODLC(vh,vv)

    Parameters
    ----------
    vh : int
        h value of MODBA
    vv : int
        v value of MODBA

    Returns
    -------
    arrLCT : np array of int
        The LC type data extracted
    '''
    import os 

    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)
    fnmLCT = dirData + 'Input/MCD12Q1_LCd_2022/MCD12Q1_LCd_500m_2022_'+strhv+'.tif'

    if os.path.exists(fnmLCT):
        arrLCT = tif2arr(fnmLCT)
        return arrLCT
    else:
        return None

def D2dcoarser(da,nbin=2,cond=None):
    ''' convert a 2D data to a coarser resolution (same rate of change at x and y)
    
    Parameters
    ----------
    da : pd DataFrame
        input 2d array, with 'x' and 'y' coords
    nbin : int
        the rate of change in size
    cond : pd DataFrame
        the optional condition to ignore some pixels in the conversion
        
    Returns
    -------
    da_cs : pd DataFrame
        the output data with coarser resolution
    '''
    # set masked data to nan
    if cond is not None:
        da = da.where(cond)
    
    # call coarsen function to do the conversion and return mean over valid pixels
    da_cs = da.coarsen(dim = {'y':nbin,'x':nbin}).mean()   
    
    return da_cs

def D2dfiner(da,nbin=10):
    ''' convert a 2d data to a finer resolution (same rate of change at x and y)
    Parameters
    ----------
    da : pd DataFrame
        input 2d array, with 'x' and 'y' coords
    nbin : int
        the rate of change in size
        
    Returns
    -------
    da_f : pd DataFrame
        the output data with finer resolution    
    '''
    import numpy as np
    import xarray as xr
    
    # use numpy to convert the data
    v = da.values.repeat(nbin,0).repeat(nbin,1)
    
    # create a pd DataFrame (note y is reversed)
    x = np.arange(da.shape[0]*nbin,dtype='int')
    y = np.flip(np.arange(da.shape[1]*nbin,dtype='int'))
    da_f = xr.DataArray(v,dims=['y','x'],coords={'x':x,'y':y})    
    
    return da_f

def getDEFM(vh,vv,usedf=False,fivekm=False,Tonly=False,ptexc=True):
    ''' Read Deforestation mask netcdf file
    Usage: arrDefM = getDEFM(yr,vh,vv)

    Parameters
    ----------
    vh : int
        h value of MODBA
    vv : int
        v value of MODBA
    usedf : bool
        return DataFrame if set to True; otherwise return numpy array
    fivekm : bool
        return 5km mask if set to True
    tonly : bool
        use the adjusted Terra only mask if set to True

    Returns
    -------
    DefM : np array of bool
        The 500m fractional DefM mask
    '''
    import os
    import xarray as xr

    # get file name
    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)

    if ptexc:
        dirDefM = dirData+'Input/MaskDefonly_2022/'
    else:
        dirDefM = dirData+'Input/MaskDef_2022/'

    if Tonly:
        strnc = '_T.nc'
    else:
        strnc = '.nc'
    fnm = dirDefM+'Def_2022_'+strhv+strnc

    # extract the data
    if os.path.exists(fnm):

        # read DefM data
        DefM = xr.open_dataarray(fnm)

        # if we use fivekm option (mask applies to whole 5km and not limited to 500m active fire pixels)
        if fivekm:
            DefM = D2dfiner(D2dcoarser(DefM,nbin=10)>0,nbin=10)

        # output dataarray or numpy array
        if usedf:
            return DefM
        else:
            return DefM.values

    else:
        return None

def getPEATM(vh,vv):
    ''' Read Deforestation mask netcdf file
    Usage: arrPeatM = getPEATM(vh,vv)

    Parameters
    ----------
    vh : int
        h value of MODBA
    vv : int
        v value of MODBA

    Returns
    -------
    PeatM : np array of bool
        The 500m fractional PeatM mask
    '''
    import os

    # get file name
    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)
    fnm = dirData+'Input/MaskPeat/Peat_'+strhv+'.tif'

    # extract the data
    if os.path.exists(fnm):
        PeatM = tif2arr(fnm).astype('bool')
        # PeatM is only valid for 60S-40N
        if (vv < 5) or (vv>14):
            PeatM[:,:] = False
        return PeatM
    else:
        return None
    
# ----------------------------------------------------------------------------------------------------
# Main steps
# ----------------------------------------------------------------------------------------------------

# Step 0: VIIRS data download and pre-processing
def read_earthdata_token():
    """
    Reads an Earthdata token from a specified text file.

    Args:
        filepath (str, optional): The path to the text file containing the token.
                                   Defaults to 'Earthdata_token.txt'.

    Returns:
        str or None: The Earthdata token read from the file, or None if the
                     file cannot be found or an error occurs during reading.
    """
    filepath=dirData + 'Input/Earthdata_token.txt'
    try:
        with open(filepath, 'r') as f:
            token = f.readline().strip()  # Read the first line and remove any leading/trailing whitespace
        return token
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{filepath}'.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return None
    
def get_edl_tokens_from_username_password():
    """
    Retrieves NASA Earthdata tokens for a given username and password.

    Args:
        username (str): The Earthdata username.
        password (str): The Earthdata password.

    Returns:
        list: A list of valid access tokens if authentication is successful,
              otherwise None. Prints an error message to the console in case of failure.
    """
    import requests

    username = input("Enter your Earthdata username: ")
    password = input("Enter your Earthdata password: ")

    test_url = "https://urs.earthdata.nasa.gov/api/users/tokens"
    response = requests.get(test_url, auth=(username, password))

    if response.status_code == 200:
        print("✅ Here is a list of all valid tokens:")
        tokens = [res['access_token'] for res in response.json()]
        return tokens
    else:
        print("❌ Authentication failed. Please check your username and password or visit https://urs.earthdata.nasa.gov/profile to generate a token.")
        return None

def get_remote_ts(url, edl_token):
    """
    Fetches unique time strings from the filenames of NetCDF (.nc) files
    listed in a remote directory's HTML listing.

    Assumes filenames follow a pattern like 'product.time_string.other_info.nc',
    where the desired time string is the second part when splitting by '.'.

    Args:
        url (str): The URL of the remote directory listing.
        edl_token (str): Your NASA Earthdata Login (EDL) bearer token.

    Returns:
        set: A set of unique time strings extracted from the filenames,
             or an empty set if no .nc files are found or an error occurs.
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "Authorization": f"Bearer {edl_token}"
    }

    # Use a set directly to store unique time strings, avoiding intermediate list
    available_times = set()

    try:
        # Make the HTTP GET request to the directory URL
        response = requests.get(url, headers=headers, timeout=10) # Added timeout
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all anchor tags (<a>), which typically contain file links
        for link in soup.find_all('a'):
            href = link.get('href')

            # Check if the link is a valid file link ending with '.nc'
            # Exclude parent directory links ('../') and current directory links ('./')
            if href and href.endswith('.nc') and not href.startswith('.'):
                try:
                    # Extract the filename from the href (handle potential paths)
                    filename = href.split('/')[-1]

                    # Split the filename by '.' and get the third part (index 2)
                    # This assumes a format like 'product.time_string.other_info.nc'
                    # Example: 'VNP14IMG_NRT.A2025139.0006.002.2025139080007.nc' -> 'A2025139'
                    parts = filename.split('.')
                    if len(parts) > 2: # Ensure there are enough parts to split
                         time_string = parts[2] # Get the second part (index 1)
                         available_times.add(time_string) # Add the extracted time to the set
                    else:
                        print(f"Warning: Filename format unexpected for '{filename}'. Skipping.")

                except IndexError:
                    # Handle cases where splitting by '.' doesn't result in enough parts
                    print(f"Warning: Could not extract time string from filename '{filename}'. Skipping.")
                except Exception as e:
                    # Catch any other unexpected errors during parsing
                    print(f"Warning: An error occurred processing link '{href}': {e}")


    except requests.exceptions.Timeout:
        print(f"Error: Request to {url} timed out.")
    except requests.exceptions.RequestException as e:
        # Handle HTTP errors (e.g., 404, 401) or network issues
        print(f"Error making request to {url}: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"An unexpected error occurred: {e}")

    return available_times

def py_wget(base_url, edl_token, output_directory=".", no_if_modified_since=True):
    """
    Downloads data from the specified URL using wget with authentication and specific parameters.

    Args:
        base_url (str): The URL of the data to download.
        edl_token (str): The Earthdata Login token for authentication.
        output_directory (str): The directory to save the downloaded data. Defaults to the current directory.
    """
    import subprocess

    # Construct the wget command
    command = [
        "wget",
        "-e", "robots=off",
        "-N", # turn on timestamping
        "-m",  # Mirror the directory structure
        "-np",  # No parent directory traversal
        "-R", ".html,.tmp",  # Reject files with extensions .html and .tmp
        "-nH",  # No host directories
        "--cut-dirs=3",  # Remove the first 3 directories from the path
        base_url,
        "--header", f"Authorization: Bearer {edl_token}",  # Add the Authorization header
        "-P", output_directory  # Set the output directory
    ]

    # Optionally add --no-if-modified-since
    if no_if_modified_since:
        command.append("--no-if-modified-since")

    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Data successfully downloaded to {output_directory}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading data: {e}")
        # sys.exit()

def checkts_VNP14IMG_daily(edl_token, yr,mo,day, NRT=False):
    """ check if VNP14IMG for all times in a day is available"""

    if NRT:
        base_url="https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG_NRT/"
    else:
        base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG/"
    url = base_url + str(yr) + '/' + strdoy(yr,mo,day) + '/'

    # download the data only if the daily data is complete
    available_times = get_remote_ts(url, edl_token)
    if len(available_times) == 0:
        return False 
    if max(available_times) == '2354':
        return True
    else:
        return False

def download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=False):
    """
    Downloads VIIRS active fire data from LANCE using wget with authentication and specific parameters.

    Args:
        base_url (str): The URL of the VIIRS active fire data to download.
        edl_token (str): The Earthdata Login token for authentication.
        output_directory (str): The directory to save the downloaded VIIRS active fire data. Defaults to the current directory.
    """

    if NRT:
        base_url="https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG_NRT/"
    else:
        base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG/"
    url = base_url + str(yr) + '/' + strdoy(yr,mo,day) + '/'

    # download the data only if the daily data is complete
    available_times = get_remote_ts(url, edl_token)
    if len(available_times) == 0:
        print(f"Warning: VNP14IMG data is incomplete for {yr}-{mo}-{day} from {base_url}.")
        return False

    if max(available_times) == '2354':
        py_wget(url, edl_token, dirData+'Input/', no_if_modified_since=no_if_modified_since)
        return True
    else:
        print(f"Warning: VNP14IMG data is incomplete for {yr}-{mo}-{day} from {base_url}.")
        return False

def check_VNP14IMG_presence(yr,mo,day):
    """ check if VNP14IMG data is available for a day """
    from glob import glob
    from datetime import datetime

    stryr = str(yr)
    doy = datetime(yr,mo,day).timetuple().tm_yday
    strdoy = str(doy).zfill(3)

    fnms = glob(dirData + 'Input/VNP14IMG/'+stryr+'/'+strdoy+'/VNP14IMG.A'+stryr+strdoy+'*.nc')

    filepresenceflag = (len(fnms) > 0)

    return filepresenceflag

def delete_files(file_list):
    """
    Deletes all files in the provided list.

    Args:
        file_list (list): List of file paths to delete.
    """
    for file_path in file_list:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def convert_VNP14IMG_to_DL(yr,mo,day,clean=False):
    """
    This function reads VNP14 netCDF files (either standard or NRT data) and extracts fire pixel data (text) into fire location data (VNP14IMGDL).
    """
    from glob import glob
    from datetime import datetime
    import xarray as xr 
    import pandas as pd
    stryr = str(yr)
    doy = datetime(yr,mo,day).timetuple().tm_yday

    # get VNP14IMG file names for the day (if not available, use VNP14IMG_NRT files)
    if check_VNP14IMG_presence(yr,mo,day):
        fnms = glob(dirData + 'Input/VNP14IMG/'+stryr+'/'+strdoy(yr,mo,day)+'/VNP14IMG.A'+stryr+strdoy(yr,mo,day)+'*.nc')
    else:
        print('No VNP14IMG data found for '+stryr+strdoy(yr,mo,day)+', use VNP14IMG_NRT instead')  
        fnms = glob(dirData + 'Input/VNP14IMG_NRT/'+stryr+'/'+strdoy(yr,mo,day)+'/VNP14IMG_NRT.A'+stryr+strdoy(yr,mo,day)+'*.nc')

    # convert image data to daily fire location (DL) data 
    dfFP = pd.DataFrame(columns=['Lon','Lat','FRP','Sample','Confidence','DNFlag'])
    for fnm in fnms:
        ds = xr.open_dataset(fnm)
        if ds.attrs['FirePix']>0:
            sFP = pd.DataFrame({'Lon':ds.FP_longitude.values,'Lat':ds.FP_latitude.values,'FRP':ds.FP_power.values,'Sample':ds.FP_sample.values,'Confidence':ds.FP_confidence.values,'DNFlag':ds.FP_day.values})
            dfFP = pd.concat([dfFP,sFP],axis=0)
    dfFP = dfFP.reset_index(drop=True)

    # save DL data to VNP14IMGDL
    dirout = dirData+'Input/VNP14IMGDL'
    mkdir(dirout)
    doy = datetime(yr,mo,day).timetuple().tm_yday
    dfFP.to_csv(dirout+'/VNP14IMGDL_'+str(yr)+str(doy).zfill(3)+'.csv')

    # delete VNP14IMG data to save space
    if clean:
        delete_files(fnms)

    return dfFP

def make_VNP14IMGDL(yr,mo,day,upd=False):
    """ for a day, download VNP14IMG netcdf and convert to location (DL) data"""
    from datetime import datetime
    import os

    # if VNP14IMGDL data already available and upd is not set to True, skip the creation
    doy = datetime(yr,mo,day).timetuple().tm_yday
    fileext = os.path.exists(dirData+'Input/VNP14IMGDL/VNP14IMGDL_'+str(yr)+str(doy).zfill(3)+'.csv')
    if fileext & (not upd):
        print('...VNP14IMGDL data already exists...')
        return True

    # otherwise, download VNPIMG or VNPIMG_NRT data and convert to fire location data

    # get the earthdata token
    edl_token = read_earthdata_token()

    # download VNP14IMG or VNP14IMG_NRT data (netcdf)   
    dlflag = False
    if checkts_VNP14IMG_daily(edl_token, yr,mo,day, NRT=False):
        print("...downloading VNP14IMG data...")
        dlflag = download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=False)
    else: # only if VNP14IMG not available, download VNP14IMG_NRT data
        if checkts_VNP14IMG_daily(edl_token, yr, mo, day, NRT=True):
            print("...downloading VNP14IMG_NRT data...")
            dlflag = download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=True)

    # convert data to text
    if dlflag:
        print("...converting VIIRS datta...")
        convert_VNP14IMG_to_DL(yr, mo, day, clean=True)    

    return dlflag

# Step 1: Record VAF at 500m resolution (MODIS 500m sinusoidal grids)
def readpreprocess_DL(yr,mo,day,IMG=True):
    ''' read and preprocess the VIIRS daily active fire location data
    '''
    # read VIIRS active fire data
    if IMG: 
        dfFC = read_VNP14IMG_NRT_daily(yr,mo,day)
        dfFC = removestaticpixels(dfFC) # filter out static pixels 
    else:
        dfFC = read_VNP14IMGML_daily(yr, mo, day)

    # add major LCT
    dfFC = adddfmjLCT(dfFC)

    # add DN flag
    dfFC = addDN(dfFC, NRT=IMG)

    return dfFC

def filter_VNP14IMGML_NRT(df_origin,scan=None, conf=None,latlon=None):
    ''' filter VIIRS active fire data

    scan (default = 0.419): float
        The along scan size for the edge grid cell. default value is for (alpha=0.5, sample=2130 or 4270); note scan value from FIRMS NRT data is size along scan, not scan angle; to not using this filter, set scan to None
    conf : list
        The confidence levels of fire pixels.
    latlon : list
        The regional box [minlat, maxlat, minlon, maxlon]. default is None
    '''
    df = df_origin.copy()
    if conf != None:  
        df = df[df['confidence'].isin(conf)]
        df = df.drop(columns='confidence')
    if scan != None:   # 0.419
        df = df[(df['scan'] <= scan)]
        df = df.drop(columns='scan')
    if latlon is not None:
        lat0, lat1, lon0, lon1 = latlon
        df = df.query('Lon >= @lon0 & Lon < @lon1 & Lat >= @lat0 & Lat < @lat1' )
    return df

def read_mjLCT():
    import xarray as xr

    # fnm_mjLCT = 'MCD12C1.A2009001.005.2011139204902.0.25x0.25.hdf'
    fnm_mjLCT = dirData+'Input/mjLCT_MCD12C1_0.25x0.25.hdf'
    da_mjLCT = xr.open_dataset(fnm_mjLCT,engine='netcdf4').rename_dims(
        {'fakeDim0':'lat','fakeDim1':'lon'}).assign_coords({'lon':G25_lons,'lat':G25_lats}).Majority_Land_Cover_Type_1
    return da_mjLCT

def get_btype(mjLCT):
    if mjLCT in [1, 2, 3, 4, 5, 6, 7, 8]:
        return 'forest'
    elif mjLCT in [9, 10]:
        return 'grass'
    elif mjLCT in [12, 14]:
        return 'crop'
    elif mjLCT in [0, 11, 13, 15, 16]:
        return 'other'
    
def adddfmjLCT(df):

    da_mjLCT = read_mjLCT()

    # Calculate the grid indices directly
    df['iLat'] = ((df['Lat'] + 90) * 4).astype(int)
    df['iLon'] = ((df['Lon'] + 180) * 4).astype(int)
    
    # Extract major land cover type using vectorized indexing
    mjLCT_array = da_mjLCT.values  # Convert DataArray to numpy array for faster indexing
    df['mjLCT'] = mjLCT_array[df['iLat'], df['iLon']]
    
    # Map mjLCT to btype
    df['btype'] = df['mjLCT'].apply(get_btype)
    
    return df

def addDN(df, NRT=True):
    if NRT:
        df['DN'] = df.pop('DNFlag').replace({1: 'day', 0: 'night'})
    else:
        df['DN'] = df.pop('DNFlag').replace({'D': 'day', 'N': 'night'})
    return df

def removestaticpixels(df):
    """ 
    Remove static pixels from VIIRS active fire data
    """
    import geopandas as gpd

    # read the static pixel mask
    gdfmask = gpd.read_file(dirData+'Input/nonvlocs.geojson')

    # Convert df to a GeoDataFrame with Point geometries
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df['Lon'], df['Lat']), 
        crs=gdfmask.crs  # Ensure the same CRS as gdfmask
    )

    # Perform a spatial join to find rows within the geometry
    gdf_in_mask = gpd.sjoin(gdf, gdfmask, how='inner', predicate='intersects')

    # Identify the rows to keep (not in gdf_in_mask)
    df_filtered = df.loc[~df.index.isin(gdf_in_mask.index)].copy()

    # # Drop the 'geometry' column if it was added
    # df_filtered.drop(columns='geometry', inplace=True)

    return df_filtered

def add_wgt_2_VNP(df, mo):
    """ add wgt column to VNP data
    """
    import pandas as pd

    # get dfwgt for the month
    dfwgt = pd.read_csv(dirData+'Input/VIIRSsplwgt_2019-2021.csv',index_col=0)
    dfwgtmidx = dfwgt.reset_index().set_index(['Sample', 'mo','DN','btype'])
    dfwgtmidxmo = dfwgtmidx.xs(mo,level='mo')

    # Use pandas' `.join` method to match and fetch 'wgt' values
    df = df.set_index(['Sample', 'DN', 'btype'])  # Temporarily set the same index as 'dfwgt'
    df['wgt'] = dfwgtmidxmo['wgt']  # Fetch 'wgt' values based on the MultiIndex
    df = df.reset_index()  # Reset index back to default

    return df

def save_VAF_1tile_day(dsFC,yr,mo,day,strhv):
    ''' save daily FC (create directory if necessary)
    '''
    import os
    from datetime import datetime

    dirFC = dirData+'Intermediate/VAF500m/'+str(yr)

    # if directory not there, create one
    if not os.path.exists(dirFC):
        os.makedirs(dirFC)

    # save data to file
    doy = datetime(yr,mo,day).timetuple().tm_yday
    fnmFC = dirFC+'/FC_'+str(doy).zfill(3)+'_'+strhv+'.nc'    
    to_netcdf(dsFC,fnmFC)

def check_VAF_1tile_day(yr,mo,day,strhv):
    ''' check if daily FC data file exists
    '''
    from datetime import datetime
    import os

    dirFC = dirData+'Intermediate/VAF500m/'+str(yr)

    # if directory not there, create one
    if not os.path.exists(dirFC):
        os.makedirs(dirFC)

    # save data to file
    doy = datetime(yr,mo,day).timetuple().tm_yday
    fnmFC = dirFC+'/FC_'+str(doy).zfill(3)+'_'+strhv+'.nc'    

    return os.path.exists(fnmFC)

def cal_VAF_1tile_1day(dfFC,vh,vv,yr,mo,day):
    ''' record and save 1 tile 1day VAF data to MODIS-500m grid cells
    '''    
    # define parameters of the tile
    strhv,xs,ys,MODgt,bb = get_tile_paras(vh,vv)

    # filter VIIRS active fire data using the bounding box of the tile
    dfFC_sub = filter_VNP14IMGML_NRT(dfFC,latlon=bb)

    if len(dfFC_sub) > 0:
        # convert pixel location to VAF numbers in each 500m grid cells 
        arrFC_month = FCpoints2arr(dfFC_sub,MODgt,xs,ys,strlon='Lon',strlat='Lat',sumcol='wgt')

        # only for tiles with nonzero active fire detections
        if arrFC_month.sum() > 0:
            # convert numerical dataarray to FC dataset
            dsFC = set_FCtile_ds(arrFC_month)

            # save FC (create directory if necessary)
            save_VAF_1tile_day(dsFC,yr,mo,day,strhv)

def recordVAF500m(yr,mo,day,dfFC, vhs=[0,35],vvs=[0,17]):
    ''' wrapper to record global (all tiles, single day) VIIRS (C1) active fire (VAF) data to MODIS-500m resolution
    '''
    # loop over all tiles and record VAF data
    if len(dfFC) > 0:
        # add sample adjustment weight
        dfFC = add_wgt_2_VNP(dfFC, mo)

        for vh in range(vhs[0],vhs[1]+1):
            for vv in range(vvs[0],vvs[1]+1):
                # print(f'Processing tile h{vh}v{vv}...')
                strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)
                if not check_VAF_1tile_day(yr,mo,day,strhv):  # if file exist, skip
                    cal_VAF_1tile_1day(dfFC,vh,vv,yr,mo,day)
    else:
        print('No data available for ', yr, mo, day)
    
    return len(dfFC)

# Step 2: Record VAF sums (GBA groups format) at 0.25deg resolution
def getVAFnc_day(yr,mo,day,vh,vv):
    ''' Read daily 500-m VAF from a precalculated netcdf file

    Parameters
    ----------
    yr : int
        year
    mo : int
        month in year 2016
    vh : int
        h value of MODBA
    vv : int
        v value of MODBA

    Returns
    -------
    FCmod : np array of bool
        The FC MOD extracted
    '''
    import xarray as xr
    import os
    from datetime import datetime

    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)

    dirFC = dirData+'Intermediate/VAF500m/'+str(yr) 

    doy = datetime(yr,mo,day).timetuple().tm_yday
    fnmFC = dirFC+'/FC_'+str(doy).zfill(3)+'_'+strhv+'.nc'    

    if os.path.exists(fnmFC):
        FC = xr.open_dataarray(fnmFC).values
        return FC
    else:
        return None

def caladd_bin_number_latlon(VAF,Tlon,Tlat,res=0.25):
    ''' Convert 2D map of VAF to 1d array of counts in each 0.25 deg grid cell
    
    Parameters
    ----------
    VAF : 2D array
        MODIS VAF image
    keynm : str
        the key name of the VAF dataframe
    Tlon : 2D array
        Lontitude values for the 2D array
    Tlat : 2D array
        Latitude values for the 2D array

    Returns
    -------
    area_all : dictionary
        updated dict of VAF dataframes (each with 'VAF','lat','lon' columns)
    '''
    import pandas as pd

    if VAF.sum() > 0:  # no change if no VIIRS active fire is detected

        # only use pixels with VAF > 0
        mask = (VAF > 0)
        df = pd.DataFrame(columns=['ilon','ilat','iVAF'])
        df['ilon'] = ((Tlon[mask]+180)/res).astype('int')
        df['ilat'] = ((Tlat[mask]+90)/res).astype('int')
        df['iVAF'] = VAF[mask]

        # use groupby to record the number of VAF in each ilon/ilat combo
        dfsize = df.groupby(['ilon','ilat']).sum()['iVAF']
        return dfsize
        
def cal_VAFp25_1tile_1day(yr,mo,day,vh,vv):
    ''' Record the VAF data to 0.25 degree grids for a 10x10 deg tile in a day.

    Parameters
    ----------
    yr : int
        year
    mo : int
        month
    vh : int
        MODIS tile h value
    vv : int
        MODIS tile v value

    Returns
    -------
    area_all : dictionary
        a dict of VAF dataframes (each with 'VAF','lat','lon' columns)
    '''
    import numpy as np

    # read 500m VAF data (produced using cal_VAF_alltiles())
    FC = getVAFnc_day(yr,mo,day,vh,vv)
    if FC is None: return None

    # read lat/lon for the tile
    Tlon,Tlat = getMODlatlon(vh,vv)

    # read LCT data
    LCT = getMODLC(vh,vv)
    if LCT is None: return None

    # read and set Peat and Defo masks
    PeatM = getPEATM(vh,vv)
    DefM = getDEFM(vh,vv,ptexc=False) # def including overlap with peat
    if (DefM is None) | (PeatM is None):     # if no peat/def mask file, set to all false array
        PeatM = np.zeros_like(LCT,dtype=bool)
        DefM = np.zeros_like(LCT,dtype=bool)
    DefonlyM = DefM & (PeatM == False)
    PeatonlyM = PeatM & (DefM == False)
    PeatdefM = (DefM & PeatM)
    NormM = ((DefM == False) & (PeatM == False))

    # calculate and record VAF for all types and biomes
    area_all = {}
    area_all['Peat'] = caladd_bin_number_latlon(FC*PeatonlyM, Tlon,Tlat)
    area_all['Defo'] = caladd_bin_number_latlon(FC*DefonlyM, Tlon,Tlat)
    area_all['PeatDefo'] = caladd_bin_number_latlon(FC*PeatdefM, Tlon,Tlat)
    for iLCT, LCnm in enumerate(LCnms):
        BiomM = ((LCT == iLCT) & NormM)
        area_all[LCnm] = caladd_bin_number_latlon(FC*BiomM,Tlon,Tlat)

    return area_all

def init_global_ds_latlon():
    ''' similar to init_global_ds, but without iFTC and iLCT
    '''

    import xarray as xr

    coords = {'lat':G25_lats,
              'lon':G25_lons}
    ds = xr.Dataset(coords = coords).fillna(0)

    return ds

def convert_MuliSer_2_da_latlon(s):
    ''' similar to convert_MuliSer_2_da, but without iFTC and iLCT
    '''
    import xarray as xr
    import numpy as np

    # record to numpy array
    arr = np.zeros((720,1440))  # iLat,iLon
    if s is not None:
        # for i, v in s.iteritems():
        for i, v in s.items():
    #         print(i,v)
            arr[i[1],i[0]] += v     # switch lat and lon

    # create xr dataarray from numpy array
    coords = {
              'lat':G25_lats,
              'lon':G25_lons }
    dims = ['lat','lon']
    da = xr.DataArray(arr,coords=coords,dims=dims)

    return da

def ds_VAF_1day_reformat(ds,yr,mo,day):
    """ reformat the VAF dataset to that consistent with the GFED5.1 EM/ dataset
    """
    import pandas as pd
    import xarray as xr
    import numpy as np
    # # read 0.25 deg grid cell area (use 2015 data)
    # grid_cell_area_km2 = get_ds_area()*1e-6
    
    # define two numpy arrays to record the daily VAF data for each biome and type
    ar_Norm_biome = np.zeros((20,1,720,1440))
    ar_type = np.zeros((5,1,720,1440))   

    # calculate total for some subgroups
    ds['Crop'] = ds['Tropic_C'] + ds['Temper_C'] + ds['Boreal_C']
    ds['Norm'] = ds['Water_W']+ds['Boreal_F']+ds['Tropic_F']+ds['Temper_F']+ds['Mosaic_FS']+ds['Tropic_SH']+ds['Temper_SH']+ds['Temper_G']+ds['Woody_SA']+ds['Open_SA']+ds['Tropic_G']+ds['Wetland_W']+ds['Urban_U']+ds['Snow_SI']+ds['Barren_B']+ds['Boreal_FS']+ds['Tundra_T']
    ds['Total'] = ds['Norm'] + ds['Crop'] + ds['Peat'] + ds['Defo'] + ds['PeatDefo']

    # clear the AF over crop biome
    ds['Tropic_C'][:,:] = 0
    ds['Temper_C'][:,:] = 0
    ds['Boreal_C'][:,:] = 0

    # extract data for different biomes in the Norm type, and different types from the input ds
    for iLCT in range(20):    
        ar_Norm_biome[iLCT,0,:,:] = ds[LCnms[iLCT]].values
    for itype,varnm in enumerate(['Norm','Crop','Defo', 'Peat', 'PeatDefo']):
        ar_type[itype,0,:,:] = ds[varnm].values

    # convert ar_Norm_biome and ar_typeto xarray dataset for each year and save
    dates = [pd.to_datetime(str(yr)+'-'+str(mo).zfill(2)+'-'+str(day).zfill(2))]
    lats = (G25_lats)[::-1]
    lons = G25_lons
    # lats = grid_cell_area_km2.lat
    # lons = grid_cell_area_km2.lon

    # map data in ds to `VAF` group in the output VAF format (extend time dim; add attrs) 
    dic_type_names = {0:'Norm', 1:'Crop', 2:'Defo', 3:'Peat', 4:'PeatDefo'}
    dic_VAF_types = {}
    for itype,typenm in dic_type_names.items():
        # print(itype,typenm)
        da_VAF_type = xr.DataArray(ar_type[itype,:,::-1,:], coords=[dates, lats, lons], dims=['time','lat', 'lon'])
        da_VAF_type.attrs['long_name']  = f"VIIRS 375m active fire pixel numbers for GFED5 {typenm} class"
        da_VAF_type.attrs['units']  = "#"
        da_VAF_type.attrs['coordinates']  = "lat lon"    
        da_VAF_type.attrs['grid_mapping']  = "latitude_longitude"
        dic_VAF_types['VAF_'+typenm] = da_VAF_type
    da_VAF_type = xr.DataArray(np.nansum(ar_type[:,:,::-1,:],axis=0), coords=[dates, lats, lons], dims=['time','lat', 'lon'])
    da_VAF_type.attrs['long_name']  = f"VIIRS 375m active fire pixel numbers"
    da_VAF_type.attrs['units']  = "# month^-1"
    da_VAF_type.attrs['coordinates']  = "lat lon"    
    da_VAF_type.attrs['grid_mapping']  = "latitude_longitude"
    dic_VAF_types['VAF_Total'] = da_VAF_type    
    ds_VAF_types = xr.Dataset(dic_VAF_types)  # convert dict to dataset

    # map data in ds to `VAF_Norm_biome` group in the output VAF format (extend time dim; add attrs)`
    list_biom_names = LCnms_full
    dic_VAF_bioms = {}
    for iLCT,LCTnm in enumerate(list_biom_names):
        da_VAF_biom = xr.DataArray(ar_Norm_biome[iLCT,:,::-1,:], coords=[dates, lats, lons], dims=['time','lat', 'lon'])
        da_VAF_biom.attrs['long_name']  = f"VIIRS 375m active fire pixel numbers for GFED5 {LCTnm}"
        da_VAF_biom.attrs['units']  = "#"
        da_VAF_biom.attrs['coordinates']  = "lat lon"    
        da_VAF_biom.attrs['grid_mapping']  = "latitude_longitude"
        dic_VAF_bioms['VAF_Norm_biome'+str(iLCT).zfill(2)] = da_VAF_biom
    ds_VAF_bioms = xr.Dataset(dic_VAF_bioms) # convert dict to dataset

    return ds_VAF_types, ds_VAF_bioms

def doVAFadjust_daily(dsAF_day, dsbAF_day, month):
    """
    Function used for adjusting daily VAF (VIIRS Active Fire) data with constrained deforestation fractions.

    This function is similar to `doVAFadjust` but operates on daily data instead of annual data.
    It adjusts the daily VAF counts based on a climatological mean fraction of deforestation and peat deforestation fires.

    Args:
        dsAF_day: Daily VIIRS active fire DataArray with dimensions (time, lat, lon).
                  Contains variables like 'VAF_Total', 'VAF_Defo', 'VAF_PeatDefo', 'VAF_Norm', 'VAF_Crop', 'VAF_Peat'.
        dsbAF_day: Daily biome-specific VIIRS active fire DataArray with dimensions (time, lat, lon).
                  Contains variables like 'VAF_Norm_biomeXX' for different biome types.
        ds_fVAF_Defo: Climatological mean fraction of VIIRS active fire in deforestation and peat deforestation
                      DataArray with dimensions (month, lat, lon).
                      Contains variables 'f_Defo' and 'f_PeatDefo'.

    Returns:
        A tuple containing two xarray DataArrays:
            - dsAF_day_new: Adjusted daily VIIRS active fire DataArray.
            - dsbAF_day_new: Adjusted daily biome-specific VIIRS active fire DataArray.
    """
    import xarray as xr

    # read climatological deforestation fractions
    ds_fVAF_Defo = xr.open_dataset(dirData+'Input/fVAF_Defo_2013-2022.nc')

    # Create copies of the original VAF data to avoid modifying the input
    dsAF_day_new = dsAF_day.copy()
    dsbAF_day_new = dsbAF_day.copy()

    # Calculate the fraction of VAF in non-deforestation (for daily data)
    fnondefo_day = (1 - dsAF_day.VAF_Defo / dsAF_day.VAF_Total - dsAF_day.VAF_PeatDefo / dsAF_day.VAF_Total)
    fnondefo_day = fnondefo_day.where(dsAF_day.VAF_Total > 0, 1)  # Where no VAF, set fnondefo_day to 1

    # Get the climatological mean fraction corresponding to the current month
    fnondefo_clim = (1 - ds_fVAF_Defo.sel(month=month).f_Defo - ds_fVAF_Defo.sel(month=month).f_PeatDefo).clip(min=0, max=1).fillna(0)
    fnondefo_clim = fnondefo_clim.where(fnondefo_clim>0,1)

    # Adjust VAF for all variables except VAF_Total
    vnms = list(dsAF_day.data_vars)
    vnms.remove('VAF_Total')
    for vnm in vnms:
        if vnm[4:] in ['Defo', 'PeatDefo']:
            # For deforestation, use VAF_Total * f_defo to get adjusted VAF_defo (same for peatdefo)
            # print(vnm)
            daVAF_vnm = dsAF_day.VAF_Total * ds_fVAF_Defo.sel(month=month)['f_' + vnm[4:]]
        elif vnm[4:] != 'Total':
            # For non-defo vegetation (Peat, Norm, Crop), use VAF(x) * fnondefo_clim/fnondefo_day to get VAF
            # print(vnm)
            daVAF_vnm = (dsAF_day[vnm] / fnondefo_day) * fnondefo_clim
           
        # Update the values in the new dataset (keep original attributes)
        dsAF_day_new[vnm] = daVAF_vnm.fillna(0)
        dsAF_day_new[vnm].attrs = dsAF_day[vnm].attrs
    

    # Recalculate VAF_Total
    dsAF_day_new['VAF_Total'] = dsAF_day_new.VAF_Norm + dsAF_day_new.VAF_Crop + dsAF_day_new.VAF_Peat + dsAF_day_new.VAF_Defo + dsAF_day_new.VAF_PeatDefo
    dsAF_day_new['VAF_Total'].attrs = dsAF_day['VAF_Total'].attrs

    # Adjust VAF for all biome variables in dsbAF_day
    vnms = list(dsbAF_day.data_vars)
    for vnm in vnms:
        dabVAF_vnm = (dsbAF_day[vnm] / fnondefo_day) * fnondefo_clim  # Use the same adjustment as for Peat, Norm, Crop

        dsbAF_day_new[vnm] = dabVAF_vnm.fillna(0)
        dsbAF_day_new[vnm].attrs = dsbAF_day[vnm].attrs

    return dsAF_day_new, dsbAF_day_new

def cal_VAFp25_alltiles_1day(yr,mo,day,vhs=[0,35],vvs=[0,17]):
    '''
    record VIIRS active fire counts (classified) for each 0.25 degree (GBA format).

    Parameters
    ----------
    yr : int
        year
    '''
    import pandas as pd
    import os

    # loop over each tile and add area to area_all
    area_keys = ['Peat','Defo','PeatDefo'] + LCnms
    area_all = None
    for vh in range(vhs[0],vhs[1]+1):
        # print(vh)
        for vv in range(vvs[0],vvs[1]+1):
            # calculate multi-index (iFTC,iLat,iLon) area sums from 1 tile
            area_1tile = cal_VAFp25_1tile_1day(yr,mo,day,vh,vv)
            if area_1tile is not None:
                if area_all is None:
                    area_all = area_1tile
                else:
                    for k in area_keys:
                        if area_1tile[k] is not None:
                            area_all[k] = pd.concat([area_all[k],area_1tile[k]])

    # save to dataset
    ds = init_global_ds_latlon()
    if area_all is not None:
        for k in area_keys:
            ds[k] = convert_MuliSer_2_da_latlon(area_all[k])

    # reformat the VAF dataset to that consistent with the GFED5 BA dataset
    ds_VAF_types, ds_VAF_bioms = ds_VAF_1day_reformat(ds,yr,mo,day)

    # adjust VAF based on deforestation climatology
    ds_VAF_types, ds_VAF_bioms = doVAFadjust_daily(ds_VAF_types, ds_VAF_bioms, mo)

    # save to netcdf
    dirout = dirData+'Intermediate/VAF/'+str(yr)
    mkdir(dirout)
    fnmout = dirout+'/VAF_'+str(yr)+'-'+str(mo).zfill(2)+'-'+str(day).zfill(2)+'.nc'
    if os.path.isfile(fnmout):
        mode = 'a'
    else:
        mode = 'w'
    to_netcdf(ds_VAF_types,fnmout,group='MOD_CMG025/VAF',mode=mode)
    to_netcdf(ds_VAF_bioms,fnmout,group='MOD_CMG025/VAF_biomes',mode='a')

# Step 3: Use pre-calculated scalar to convert VAF sums to BA sums (G16 format)    
def read_GFEDmask():
    ''' Read GFED regional mask array from the GFED mask file
    '''
    import xarray as xr
    
    arrGFED = tif2arr(dirData+'Input/GFEDregions_025d.tif')
    daGFED = xr.DataArray(arrGFED[::-1,:],coords={'lat':G25_lats,'lon':G25_lons},dims=['lat','lon'])
    
    return daGFED

def mapGFEDregdata(dfdata, daGFED, GFEDnms):
    """
    Maps regional mean values from a Series to a global DataArray using a regional mask.

    Parameters
    ----------
    dfdata : pandas.Series
        DataFrame containing regional mean values. The index should be the region names
        (e.g., 'AUST', 'EURO'), and the series should contain the values to map.
    daGFED : xarray.DataArray
        2D DataArray representing the global GFED mask, where each grid cell's value is an integer
        corresponding to a region index. This mask defines the spatial distribution of the regions.
    GFEDnms : list of str
        List of region names corresponding to the region indices in `daGFED`. The order of the list
        should match the indices in `daGFED` (e.g., GFEDnms[0] corresponds to region index 0).

    Returns
    -------
    xarray.DataArray
        A new DataArray with the same dimensions as `daGFED`, but with the values from `dfdata`
        mapped according to the region indices in `daGFED`. Cells corresponding to regions not in 
        `dfdata` will be filled with NaN.

    Example
    -------
    dadata = mapGFEDregdata(dfdata, daGFED, GFEDnms)
    """
    import xarray as xr
    import numpy as np

    # Create a mapping from region index to mean values using the GFEDnms array.
    # If a region name in GFEDnms is found in dfdata, map its index to the corresponding value.
    # Otherwise, map the index to NaN.
    region_index_to_value = {
        i: dfdata.loc[GFEDnms[i]] if GFEDnms[i] in dfdata.index else np.nan
        for i in range(len(GFEDnms))
    }

    # Vectorize the mapping function to apply it across the entire DataArray efficiently.
    vectorized_map = np.vectorize(region_index_to_value.get)

    # Apply the vectorized mapping function to the DataArray `daGFED` using xr.apply_ufunc.
    # This replaces region indices in `daGFED` with the corresponding values from `dfdata`.
    dadata = xr.apply_ufunc(vectorized_map, daGFED, dask='parallelized', output_dtypes=[float])
    
    return dadata

def read_VAFp25_alltiles_1day(yr, mo, day):
    """ read daily VIIRS AF data at 0.25 deg
    """
    import xarray as xr 

    fnmVAF = dirData+'Intermediate/VAF/'+str(yr)+'/VAF_'+strymd(yr,mo,day)+'.nc'
    ds = xr.open_dataset(fnmVAF,group='MOD_CMG025/VAF')
    dsb = xr.open_dataset(fnmVAF,group='MOD_CMG025/VAF_biomes')
    ds_coord = xr.open_dataset(fnmVAF,group='MOD_CMG025')
    dsAF_yr = ds.assign_coords(ds_coord.coords)
    dsbAF_yr = dsb.assign_coords(ds_coord.coords)

    return dsAF_yr, dsbAF_yr

def read_VFA():
    """ read presaed VFA tables 
    """
    import pandas as pd
    dfVFA = pd.read_csv(dirData + 'Input/GFED51EFAc_regtp.csv',index_col=0)
    return dfVFA

def remapBAclass(dsp25,damaskp25TropFor, damaskp25TempFor, damaskp25Other, FTC0103):
    """ convert the original GFED5.1 BA dataset (biom + type, in km2) to the GFED5.1 16-class dataarray (in m2)

    - The input dataset `dsp25` is read from dsp25 = xr.open_mfdataset(dirGFED51BA+'BA'+str(yr)+'??.nc').load()
    - damaskp25TropFor, damaskp25TempFor, damaskp25Other are the 0.25 deg LCT fractions
    - FTC0103 is the forest fraction averaged over 2001-2003, read from readFTC0103()
    """
    import xarray as xr

    # initialize yearly BA dataarray 16class BA data
    daBA_16class_yr = xr.DataArray(
        dims=['lct','time','lat','lon'],
        coords={'lct':EMLCnms,'lat':G25_lats,'lon':G25_lons,'time':dsp25.time})

    # map GFED5.1 original BA classes to GFED5.1 EM/BA 16 LCTs (m2)
    # 18 -> 1 'Tundra'
    daBA_16class_yr.loc[{'lct':'Tundra'}] = dsp25.Norm.loc[{'iLCT':18}] 
    # 17 -> 2 'Sparse boreal forest'
    daBA_16class_yr.loc[{'lct':'Sparse boreal forest'}] = dsp25.Norm.loc[{'iLCT':[17]}].sum(dim='iLCT')
    # 1 -> 3 'Boreal forest'
    daBA_16class_yr.loc[{'lct':'Boreal forest'}] = dsp25.Norm.loc[{'iLCT':[1]}].sum(dim='iLCT')
    # 7 -> 4 'Temperate grassland'
    daBA_16class_yr.loc[{'lct':'Temperate grassland'}] = dsp25.Norm.loc[{'iLCT':[7]}].sum(dim='iLCT')
    # 6 -> 5 'Temperate shrubland'
    daBA_16class_yr.loc[{'lct':'Temperate shrubland'}] = dsp25.Norm.loc[{'iLCT':[6]}].sum(dim='iLCT')
    # 4 -> 6 'Temperate mosaic'
    daBA_16class_yr.loc[{'lct':'Temperate mosaic'}] = dsp25.Norm.loc[{'iLCT':[4]}].sum(dim='iLCT')
    # 2 (TempFor) + Defonly (TempFor) -> 7 'Temperate forest'
    daBA_16class_yr.loc[{'lct':'Temperate forest'}] = dsp25.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TempFor * FTC0103 + dsp25.Defonly * damaskp25TempFor + dsp25.Norm.loc[{'iLCT':[3]}].sum(dim='iLCT')
    # 10 -> 8 'Tropical grassland'
    daBA_16class_yr.loc[{'lct':'Tropical grassland'}] = dsp25.Norm.loc[{'iLCT':[10]}].sum(dim='iLCT')
    # 5 -> 9 'Tropical shrubland'
    daBA_16class_yr.loc[{'lct':'Tropical shrubland'}] = dsp25.Norm.loc[{'iLCT':[5]}].sum(dim='iLCT')
    # 9 -> 10 'Open savanna'
    daBA_16class_yr.loc[{'lct':'Open savanna'}] = dsp25.Norm.loc[{'iLCT':[9]}].sum(dim='iLCT')
    # 8 + 2 (Other) + Defonly (Other) -> 11 'Woody savanna'
    daBA_16class_yr.loc[{'lct':'Woody savanna'}] = dsp25.Norm.loc[{'iLCT':[8]}].sum(dim='iLCT') + dsp25.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25Other + dsp25.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TropFor * (1-FTC0103) +  dsp25.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TempFor * (1-FTC0103) + dsp25.Defonly * damaskp25Other 
    # 2 (TropFor) -> 12 'Tropical forest' 
    daBA_16class_yr.loc[{'lct':'Tropical forest'}] = dsp25.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TropFor * FTC0103

    # 0, 13, 15, 16 -> 13 'Other'
    daBA_16class_yr.loc[{'lct':'Other'}] = dsp25.Norm.loc[{'iLCT':[0,13,15,16]}].sum(dim='iLCT')
    # Crop -> 14 'Cropland'
    daBA_16class_yr.loc[{'lct':'Cropland'}] = dsp25.Crop
    # 11 + Peat + PeatDefo -> 15 'Peat'
    daBA_16class_yr.loc[{'lct':'Peat'}] = dsp25.Norm.loc[{'iLCT':[11]}].sum(dim='iLCT') + dsp25.Peatonly  + dsp25.Peatdef 
    # Defonly (TropFor) -> 16 'Deforestation'
    daBA_16class_yr.loc[{'lct':'Deforestation'}] = dsp25.Defonly * damaskp25TropFor

    # fill in N/A and convert from km2 to m2 (since FC is in g C/m2)
    daBA_16class_yr = daBA_16class_yr.fillna(0)* 1e6  # km2 --> m2

    return daBA_16class_yr

def cal_BA_scled_day(yr, mo, day):
    """ Use EFA scalars to convet daily VIIRs active fire counts to BA; convert to G16 format and save to netcdf
    """
    import xarray as xr
    import numpy as np

    # read GFED mask
    daGFED = read_GFEDmask()

    # read daily VIIRS AF data (0.25 deg, GBA format)
    dsAF_yr, dsbAF_yr = read_VAFp25_alltiles_1day(yr, mo, day)

    # read presaved AFA tables (GBA format)
    dfVFA = read_VFA()

    # scale AF to BA (Norm) and record to a da
    t = np.datetime64(strymd(yr,mo,day))
    daBA_Norm = xr.DataArray(dims=['time','iLCT','lat','lon'],coords={
        'time':[t], 'iLCT':np.arange(20, dtype='int'), 'lat':G25_lats, 'lon':G25_lons
    })
    for ib in range(20):
        if ib not in [12,14,19]: # skip crop biomes

            # extract daAF_ib
            daAF_ib = dsbAF_yr['VAF_Norm_biome'+str(ib).zfill(2)].sel(time=t)  # monthly AF

            # extract daAFA_ib for scaling used in different options
            dfVFAftp = dfVFA.loc[LCnms[ib]][:-1]  # ignore global
            daAFA_ib = mapGFEDregdata(dfVFAftp, daGFED, GFEDnms)

            # do the conversion for ib
            daBA_Norm.loc[{'iLCT':ib,'time':t}] = (daAFA_ib * daAF_ib)

    # record BA(Norm)
    daBA_Norm = daBA_Norm.where((daBA_Norm>0) & (daBA_Norm!=np.inf),0)
    ds = xr.Dataset({'Norm':daBA_Norm})
    
    # scale AF to BA (Crop, Peat, Defo, PeatDefo)
    BAbiomstr = {'Crop':'Crop', 'Peat':'Peatonly', 'Defo':'Defonly', 'PeatDefo':'Peatdef'}
    for biom in ['Crop', 'Peat', 'Defo', 'PeatDefo']:
        daBA_biom = xr.DataArray(dims=['time','lat','lon'],coords={
            'time':[t], 'lat':G25_lats, 'lon':G25_lons})
        daAF_mo = dsAF_yr['VAF_'+biom].sel(time=t)

        # extract daAFA_ib for scaling used in different options
        dfVFAftp = dfVFA.loc[biom][:-1]  # ignore global
        daAFA_mo = mapGFEDregdata(dfVFAftp, daGFED, GFEDnms)
        daBA_biom.loc[{'time':t}] = (daAFA_mo * daAF_mo)
        daBA_biom = daBA_biom.where((daBA_biom>0) & (daBA_biom!=np.inf),0)
        ds[BAbiomstr[biom]] = daBA_biom

    # calculate sum
    daBA_Total = daBA_Norm.sum(dim='iLCT') + ds['Crop'] + ds['Peatonly'] + ds['Defonly'] + ds['Peatdef']
    ds['Total'] = daBA_Total

    # convert BA data (type/biome, km2) to 16-class format (m2)
    damaskp25TropFor, damaskp25TempFor, damaskp25Other = readformasks() # read 0.25deg tropical and temperate forest masks
    FTC0103 = readFTC0103()  # read FTC0103
    da_16class = remapBAclass(ds,damaskp25TropFor, damaskp25TempFor, damaskp25Other, FTC0103).transpose('lct', 'time', 'lat', 'lon').sortby('lat', ascending=True)  # make sure follows ('lct', 'time', 'lat', 'lon') sequence and lat from -90 to 90
    ds_16class_yr = da_16class.to_dataset(name='BA')

    # # add G16 attributes
    # ds_16class_yr = add_BA_attrs(ds_16class_yr)

    # save 16-class BA data
    dirout = dirData+'Intermediate/BA/'+str(yr)
    mkdir(dirout)
    fnmout = dirout+'/BA_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_16class_yr, fnmout)

# Step 4: Use pre-calculated scalar to convert BA sums to EM sums (G16 format)
def readformasks():
    ''' Read annual .25x.25 GFED5 LCT (.tif) to create masks for tropical forest, temperate forest, and others
        The output masks can be used to mapping GFED5.1 original BA classes (biomes+types) to BA-16 classes.
    '''
    import rasterio as rio
    import xarray as xr

    # read GFED5 LCT tif data
    tifdata = rio.open(dirData+'Input/MCD12Q1_LCd_025d_2022.tif')
    LCTp25data = tifdata.read(1)  # numpy array, 0-19
    maskp25TropFor = (LCTp25data == 2)
    maskp25TempFor = (LCTp25data == 3) | (LCTp25data == 4) # 'Forest temperate','Sparse temperate mosaic'

    # create mask dataarrays
    damaskp25TropFor = xr.DataArray(maskp25TropFor[::-1,:], dims=['lat', 'lon'], coords={'lat':G25_lats,'lon':G25_lons})
    damaskp25TempFor = xr.DataArray(maskp25TempFor[::-1,:], dims=['lat', 'lon'], coords={'lat':G25_lats,'lon':G25_lons})
    damaskp25Other = (~damaskp25TempFor & ~damaskp25TropFor)

    return damaskp25TropFor, damaskp25TempFor, damaskp25Other

def readFTC0103():
    """ read the forest fraction averaged over 2001-2003
    The data were produced by Guido van der Werf; It does not represents the actual forest fraction, but the fraction of the tropical forest burned area that can be kept in the remapped 'tropical forest' and 'temperate forest' (the rest fraction goes to 'woody savanna')

    
    """
    import numpy as np
    import xarray as xr

    fnm = dirData + 'Input/fraction_forest_BA.txt'
    data = np.loadtxt(fnm, converters=float)

    FTC0103 = xr.DataArray(data[::-1,:],dims=['lat','lon'],coords={'lat':G25_lats,'lon':G25_lons})

    return FTC0103

def read_FC():
    """ read presaed FC tables 
    """
    import pandas as pd
    dfFC = pd.read_csv(dirData + 'Input/GFED51FC_regtp.csv',index_col=0)
    return dfFC

def cal_EM_scled_day(yr, mo, day):
    """ Use FC scalars to convet daily VIIRS BA to EM; keep G16 format and save to netcdf
    """
    import xarray as xr 
    import numpy as np

    # read annual GFED5.1 original BA (km2) or GFED5.1 BAfromVAF (km2)
    dsBA_16class = read_BA(yr, mo, day)
    if dsBA_16class is None:
        return
    daBA_16class = dsBA_16class.BA

    # read GFEDmask and GFEDnms
    daGFED = read_GFEDmask()

    # read FC tables for different options
    daFC = read_FC()

    # scale the BA to EM scaling for each month
    t = np.datetime64(strymd(yr,mo,day))
    daEM_fromBA = xr.DataArray(dims=['lct','time','lat','lon'],coords={
        'lct':EMLCnms,'time':[t],'lat':G25_lats, 'lon':G25_lons
    })
    
    for _, tp in enumerate(EMLCnms):
        dfFCftp = daFC.loc[tp][:-1]  # ignore global
        daFC_ib = mapGFEDregdata(dfFCftp, daGFED, GFEDnms)

        # extract monthly BA for tp and do the scaling
        daBA_ib = daBA_16class.sel(lct=tp).sel(time=t)  
        daEM_fromBA.loc[{'lct':tp,'time':t}] = (daBA_ib * daFC_ib)


    # convert to ds
    ds_16class_yr = daEM_fromBA.to_dataset(name='EM')

    # # add G16 attributes
    # ds_16class_yr = add_EM_attrs(ds_16class_yr)

    # save output
    dirout = dirData+'Intermediate/EM/'+str(yr)
    mkdir(dirout)
    fnmout = dirout+'/EM_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_16class_yr, fnmout)

# Step 5: Derive GFED5eNRTeco (16-class combined data VAF + BA + EM)
def add_GFED5eco_attrs(ds):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ds.attrs['long_name'] = 'Global Fire Emissions Database v5eNRT ecosystem'
    ds.attrs['standard_name'] = 'GFED5eNRTeco'
    ds.attrs['creation_date'] = now
    ds.attrs['frequency'] = 'day'
    ds.attrs['grid'] = '0.25x0.25 degree latxlon grid'  # change 1x1 for pre-MODIS data
    ds.attrs['institution'] = 'University of California, Irvine'
    ds.attrs['license'] = 'Data in this file is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License(https://creativecommons.org/licenses/)' 
    ds.attrs['nominal_resolution'] = '0.25x0.25 degree'  # change to 1x1 degree for pre-MODIS data 
    ds.attrs['region'] = 'global'
    ds.attrs['contact'] = 'Yang Chen (yang.chen@uci.edu)'

    ds.lat.attrs['units'] = "degrees_north"
    ds.lat.attrs['axis'] = "Y" 
    ds.lat.attrs['long_name'] = "latitude" 
    ds.lat.attrs['standard_name'] = "lat" 

    ds.lon.attrs['units'] = "degrees_east"
    ds.lon.attrs['axis'] = "X" 
    ds.lon.attrs['long_name'] = "longitude" 
    ds.lon.attrs['standard_name'] = "lon" 

    ds.lct.attrs['long_name'] = "ecosystem land cover type"
    ds.lct.attrs['standard_name'] = "lct"
    ds.lct.attrs['source'] = "GFED5 reclassified types (16-class), Table 2 of van der Werf et al. (2025)"
    return ds

def add_BA_attrs(ds):
    ds.BA.attrs['long_name'] = "burned area"
    ds.BA.attrs['standard_name'] = "BA"
    ds.BA.attrs['unit'] = "m2/day"
    return ds 

def add_EM_attrs(ds):
    ds.EM.attrs['long_name'] = "emissions"
    ds.EM.attrs['standard_name'] = "EM"
    ds.EM.attrs['unit'] = "g C/day"
    return ds 

def add_VAF_attrs(ds):
    ds.VAF.attrs['long_name'] = "VIIRS 375m active fire counts"
    ds.VAF.attrs['standard_name'] = "VAF"
    ds.VAF.attrs['unit'] = "#/day"
    return ds 

def getVAF16class(yr,mo,day):
    """ convert the original GFED5.1 BA dataset (biom + type, in km2) to the GFED5.1 16-class dataarray (in m2)

    - The input dataset `dsp25` is read from dsp25 = xr.open_mfdataset(dirGFED51BA+'BA'+str(yr)+'??.nc').load()
    - damaskp25TropFor, damaskp25TempFor, damaskp25Other are the 0.25 deg LCT fractions
    - FTC0103 is the forest fraction averaged over 2001-2003, read from readFTC0103()
    """
    import xarray as xr

    # read daily VIIRS AF data (0.25 deg, GBA-2groups format)
    dsAF_yr, dsbAF_yr = read_VAFp25_alltiles_1day(yr, mo, day)

    # convert VAF from GBA-2groups to GBA format
    t = np.datetime64(strymd(yr,mo,day))
    daAF_Norm = xr.DataArray(dims=['time','iLCT','lat','lon'],coords={
        'time':[t], 'iLCT':np.arange(20, dtype='int'), 'lat':dsAF_yr.lat, 'lon':dsAF_yr.lon
    })
    for ib in range(20):
        if ib not in [12,14,19]: # skip crop biome
            daAF_Norm.loc[{'iLCT':ib,'time':t}] = dsbAF_yr['VAF_Norm_biome'+str(ib).zfill(2)].sel(time=t)
    daAF_Norm = daAF_Norm.where((daAF_Norm>0) & (daAF_Norm!=np.inf),0)
    ds = xr.Dataset({'Norm':daAF_Norm})
    BAbiomstr = {'Crop':'Crop', 'Peat':'Peatonly', 'Defo':'Defonly', 'PeatDefo':'Peatdef'}
    for biom in ['Crop', 'Peat', 'Defo', 'PeatDefo']:
        daAF_biom = xr.DataArray(dims=['time','lat','lon'],coords={'time':[t], 'lat':dsAF_yr.lat, 'lon':dsAF_yr.lon})
        daAF_biom.loc[{'time':t}] = dsAF_yr['VAF_'+biom].sel(time=t)
        daAF_biom = daAF_biom.where((daAF_biom>0) & (daAF_biom!=np.inf),0)
        ds[BAbiomstr[biom]] = daAF_biom
    daAF_Total = daAF_Norm.sum(dim='iLCT') + ds['Crop'] + ds['Peatonly'] + ds['Defonly'] + ds['Peatdef']
    ds['Total'] = daAF_Total

    # read data for format mapping
    damaskp25TropFor, damaskp25TempFor, damaskp25Other = readformasks() # read 0.25deg tropical and temperate forest masks
    FTC0103 = readFTC0103()  # read FTC0103

    # initialize yearly VAF dataarray 16class data
    daVAF_16class = xr.DataArray(
        dims=['lct','time','lat','lon'],
        coords={'lct':EMLCnms,'lat':ds.lat,'lon':ds.lon,'time':ds.time})

    # map VAF values from GBA to G16 format
    # 18 -> 1 'Tundra'
    daVAF_16class.loc[{'lct':'Tundra'}] = ds.Norm.loc[{'iLCT':18}] 
    # 17 -> 2 'Sparse boreal forest'
    daVAF_16class.loc[{'lct':'Sparse boreal forest'}] = ds.Norm.loc[{'iLCT':[17]}].sum(dim='iLCT')
    # 1 -> 3 'Boreal forest'
    daVAF_16class.loc[{'lct':'Boreal forest'}] = ds.Norm.loc[{'iLCT':[1]}].sum(dim='iLCT')
    # 7 -> 4 'Temperate grassland'
    daVAF_16class.loc[{'lct':'Temperate grassland'}] = ds.Norm.loc[{'iLCT':[7]}].sum(dim='iLCT')
    # 6 -> 5 'Temperate shrubland'
    daVAF_16class.loc[{'lct':'Temperate shrubland'}] = ds.Norm.loc[{'iLCT':[6]}].sum(dim='iLCT')
    # 4 -> 6 'Temperate mosaic'
    daVAF_16class.loc[{'lct':'Temperate mosaic'}] = ds.Norm.loc[{'iLCT':[4]}].sum(dim='iLCT')
    # 2 (TempFor) + Defonly (TempFor) -> 7 'Temperate forest'
    daVAF_16class.loc[{'lct':'Temperate forest'}] = ds.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TempFor * FTC0103 + ds.Defonly * damaskp25TempFor + ds.Norm.loc[{'iLCT':[3]}].sum(dim='iLCT')
    # 10 -> 8 'Tropical grassland'
    daVAF_16class.loc[{'lct':'Tropical grassland'}] = ds.Norm.loc[{'iLCT':[10]}].sum(dim='iLCT')
    # 5 -> 9 'Tropical shrubland'
    daVAF_16class.loc[{'lct':'Tropical shrubland'}] = ds.Norm.loc[{'iLCT':[5]}].sum(dim='iLCT')
    # 9 -> 10 'Open savanna'
    daVAF_16class.loc[{'lct':'Open savanna'}] = ds.Norm.loc[{'iLCT':[9]}].sum(dim='iLCT')
    # 8 + 2 (Other) + Defonly (Other) -> 11 'Woody savanna'
    daVAF_16class.loc[{'lct':'Woody savanna'}] = ds.Norm.loc[{'iLCT':[8]}].sum(dim='iLCT') + ds.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25Other + ds.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TropFor * (1-FTC0103) +  ds.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TempFor * (1-FTC0103) + ds.Defonly * damaskp25Other 
    # 2 (TropFor) -> 12 'Tropical forest' 
    daVAF_16class.loc[{'lct':'Tropical forest'}] = ds.Norm.loc[{'iLCT':[2]}].sum(dim='iLCT') * damaskp25TropFor * FTC0103
    # 0, 13, 15, 16 -> 13 'Other'
    daVAF_16class.loc[{'lct':'Other'}] = ds.Norm.loc[{'iLCT':[0,13,15,16]}].sum(dim='iLCT')
    # Crop -> 14 'Cropland'
    daVAF_16class.loc[{'lct':'Cropland'}] = ds.Crop
    # 11 + Peat + PeatDefo -> 15 'Peat'
    daVAF_16class.loc[{'lct':'Peat'}] = ds.Norm.loc[{'iLCT':[11]}].sum(dim='iLCT') + ds.Peatonly  + ds.Peatdef 
    # Defonly (TropFor) -> 16 'Deforestation'
    daVAF_16class.loc[{'lct':'Deforestation'}] = ds.Defonly * damaskp25TropFor

    # fill in N/A
    daVAF_16class = daVAF_16class.fillna(0).transpose('lct', 'time', 'lat', 'lon').sortby('lat', ascending=True)

    return daVAF_16class

def make_GFED5eco(yr,mo,day):
    """ createt GFED5e ecosystem 16-class data, which contains three layers
     - VAF: VIIRS active fire counts
     - BA: burned area
     - EM: emissions
    """
    # read 16-class BA data as the base
    ds_combined = read_BA(yr, mo, day)

    # add global attributes
    ds_combined = add_GFED5eco_attrs(ds_combined)

    # add BA attributes
    ds_combined = add_BA_attrs(ds_combined)

    # read 16-class EM data and add to ds_combined
    dsEM_16class = read_EM(yr, mo, day)
    ds_combined['EM'] = dsEM_16class['EM']

    # add EM attributes
    ds_combined = add_EM_attrs(ds_combined)

    # read GBA VAF data
    daVAF_16class = getVAF16class(yr,mo,day)
    ds_combined['VAF'] = daVAF_16class

    # add VAF attributes
    ds_combined = add_VAF_attrs(ds_combined)

    # save output
    dirout = dirData+'Output/'+str(yr)
    mkdir(dirout)
    fnmout = dirout+'/GFED5eNRTeco_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_combined, fnmout)

# Step 6: Derive GFED5eNRTspe (species emissions
def map16to6(EM16):
    """ Convert fire emissions (C) from 16 types to 6 types
    """

    # set mapping from 16 types to 6 types
    class6 = ['Savanna and grassland','Boreal forest','Temperate forest','Deforestation & degradation','Peatlands','Agriculture']

    class16 = ['Tundra','Sparse boreal forest','Boreal forest','Temperate grassland','Temperate shrubland','Temperate mosaic','Temperate forest','Tropical grassland','Tropical shrubland','Open savanna','Woody savanna','Tropical forest','Other','Cropland','Peat','Deforestation']

    classmap = {'Tundra':'Boreal forest', 'Sparse boreal forest':'Boreal forest', 'Boreal forest':'Boreal forest',
        'Temperate grassland':'Savanna and grassland', 'Temperate shrubland':'Savanna and grassland', 'Temperate mosaic':'Savanna and grassland',
        'Temperate forest':'Temperate forest', 'Tropical grassland':'Savanna and grassland', 'Tropical shrubland':'Savanna and grassland',
        'Open savanna':'Savanna and grassland', 'Woody savanna':'Savanna and grassland', 'Tropical forest':'Deforestation & degradation', 'Other':'Savanna and grassland', 'Cropland':'Agriculture',
        'Peat':'Peatlands', 'Deforestation':'Deforestation & degradation'}

    # Map lct index values to corresponding aggregated categories
    agct_mapping = np.array([classmap[lct] for lct in class16]) 

    # Create an empty DataArray for aggregated data
    EM6 = xr.DataArray(
        np.zeros((1, len(class6), 720, 1440)),  # Same spatial resolution, but lct is replaced by agct
        dims=["time", "ftype", "lat", "lon"],
        coords={"time": EM16.time, "ftype": class6, "lat": EM16.lat, "lon": EM16.lon}
    )

    # Sum EM over original lct categories, grouping them by agct
    for i, agct in enumerate(class6):
        mask = agct_mapping == agct  # Get indices of lct that map to the current agct
        EM6.loc[{"ftype": agct}] = EM16.sel(lct=mask).sum(dim="lct")

    return EM6.transpose('ftype', 'time', 'lat', 'lon').sortby('lat', ascending=True)

def add_GFED5spe_attrs(ds):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ds.attrs['long_name'] = 'Global Fire Emissions Database v5eNRT species emissions'
    ds.attrs['standard_name'] = 'GFED5eNRTspe'
    ds.attrs['unit'] = "g/day"
    ds.attrs['creation_date'] = now
    ds.attrs['frequency'] = 'day'
    ds.attrs['grid'] = '0.25x0.25 degree latxlon grid'  # change 1x1 for pre-MODIS data
    ds.attrs['species'] = 'See van der Werf et al. (2025) for a complete list of species'
    ds.attrs['institution'] = 'University of California, Irvine'
    ds.attrs['license'] = 'Data in this file is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License(https://creativecommons.org/licenses/)' 
    ds.attrs['nominal_resolution'] = '0.25x0.25 degree'  # change to 1x1 degree for pre-MODIS data 
    ds.attrs['region'] = 'global'
    ds.attrs['contact'] = 'Yang Chen (yang.chen@uci.edu)'

    ds.lat.attrs['units'] = "degrees_north"
    ds.lat.attrs['axis'] = "Y" 
    ds.lat.attrs['long_name'] = "latitude" 
    ds.lat.attrs['standard_name'] = "lat" 

    ds.lon.attrs['units'] = "degrees_east"
    ds.lon.attrs['axis'] = "X" 
    ds.lon.attrs['long_name'] = "longitude" 
    ds.lon.attrs['standard_name'] = "lon" 

    return ds

def make_EMspecies(yr,mo,day):
    """ save 6-class EM files
    """

    # read emission data from GFED5eNRT ecosystem
    EM16 = read_GFED5eco(yr,mo,day).EM 

    # map EM to 6-classes
    EM6 = map16to6(EM16)
    ds_EM6_species = xr.Dataset({'Carbon': EM6})

    # Load emission factors (EF, g/kg C) from CSV
    dfEF = pd.read_csv(dirData+'Input/GFED51_EF.csv', index_col=0)

    # calculate grid cell sum emissions for each species
    DMC = dfEF.loc['DMC',]/100  # carbon fraction in dry matter
    sps = dfEF.index[:-1] # list of species (skip DMC)
    for sp in sps:
        # read EF for each species
        EFsp = dfEF.loc[sp]

        # convert EF to g / g C and ensure alignment with EM6 ftype coordinates
        EFcarbon = xr.DataArray(EFsp/DMC, dims=["ftype"], coords={"ftype": EM6.ftype}) / 1e3  # g / gC 

        # calculate emissions for each species
        ds_EM6_species[sp] = EM6 * EFcarbon  # g / day    
    ds_EM6_species_sum = ds_EM6_species.sum(dim='ftype')

    # add attributes for species file
    ds_EM6_species_sum= add_GFED5spe_attrs(ds_EM6_species_sum)

    # save output (only the sum emissions)
    dirout = dirData+'Output/'+str(yr)
    mkdir(dirout)
    fnmout = dirout+'/GFED5eNRTspe_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_EM6_species_sum, fnmout)

    return ds_EM6_species

# Optional steps : Use cloud fraction to scale the BA or EM (currently not used)
def doCFscl(da, daCF):
    ''' scale dataarray `da` by cloud fraction `daCF`
    daCF should have same format with daCFmean
    '''
    import xarray as xr

    daCFmean = xr.open_dataarray(dirData+'Input/CFmean_2014-2023.nc')

    scler = (1-daCFmean)/(1-daCF)

    da_scl = da * scler

    return da_scl

# ----------------------------------------------------------------------------------------------------
# Creating figures showing cumulative regional sum (in comparison to climatological data)
# ----------------------------------------------------------------------------------------------------

def readGFED5eco(yr, mo=None, day=None, vnm='EM'):
    """
    Load a variable from GFED5.1 NRT NetCDF files for a specified year, month, and/or day.

    Parameters
    ----------
    yr : int
        Year of data to read (e.g., 2025).
    mo : int, optional
        Month of data to read (1-12). If None, loads all months for the year.
    day : int, optional
        Day of month to read (1-31). If None, loads all days for the month/year.
    vnm : str, optional
        Variable name to extract from the dataset (default is 'EM').

    Returns
    -------
    xarray.DataArray
        DataArray containing the requested variable for the specified time period.

    Notes
    -----
    - If only `yr` is provided, loads all daily files for the year.
    - If `yr` and `mo` are provided, loads all daily files for that month.
    - If `yr`, `mo`, and `day` are provided, loads the specific daily file.
    - Requires `dirData` to be defined globally.
    
    Example use
    -----------
    EM20250525 = readGFED5eco(2025, 5, 25, 'EM')
    EM202505 = readGFED5eco(yr=2025, mo=5, vnm='EM')
    EM2025 = readGFED5eco(yr=2025, vnm='EM')
    """
    import xarray as xr
    from glob import glob

    if day is not None:
        fnm = dirData + 'Output/' + str(yr) + '/GFED5eNRTeco_' + str(yr) + '-' + str(mo).zfill(2) + '-' + str(day).zfill(2) + '.nc'
        ds = xr.open_dataset(fnm)
    elif mo is not None:
        fnms = glob(dirData + 'Output/' + str(yr) + '/GFED5eNRTeco_' + str(yr) + '-' + str(mo).zfill(2) + '-*.nc')
        ds = xr.open_mfdataset(fnms)
    else:
        fnms = glob(dirData + 'Output/' + str(yr) + '/GFED5eNRTeco_' + str(yr) + '-*.nc')
        ds = xr.open_mfdataset(fnms)

    da = ds[vnm]
    return da

def readcumudata(yr, vnm='EM'):
    """
    Read cumulative emissions data for a given year and variable name.
    
    Parameters
    ----------
    yr : int
        Year of data to read (e.g., 2025).
    vnm : str, optional
        Variable name to extract from the dataset (default is 'EM').
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing cumulative emissions data for the specified year.
    """
    import pandas as pd
    import os
    
    fnm = dirData + 'Output/' + str(yr) + '/GFED5eNRTcumu' + vnm + '_' + str(yr) + '.csv'
    if not os.path.exists(fnm):
        return None
    else:
        df = pd.read_csv(fnm, index_col=0, parse_dates=True)
    
    return df

def doGFEDregsum(da,yr,mo, day):
    import pandas as pd
    import datetime
    
    daGFED = read_GFEDmask()
    
    sums = {}
    sums['globe'] = da.sum().item()
    for region_id, region_name in enumerate(GFEDnms[1:]):
        # Mask the data array with the current region
        region_mask = daGFED == (region_id + 1)
        regional_da = da.where(region_mask, drop=True)
        
        # Calculate the sum for the masked region
        sums[region_name] = regional_da.sum().item()
        
    time_value = datetime.date(yr,mo, day)  # Example date, adjust as needed
    df_day = pd.DataFrame(sums, index=[time_value])
    
    return df_day

def updateGFEDregsum(df, df_day, yr, vnm):
    """
    Update the DataFrame with new regional sums for a specific day.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to update with new regional sums.
    df_day : pandas.DataFrame
        DataFrame containing the new regional sums for the specified day.
    yr : int, optional
        Year of the data (default is global variable yr).
    vnm : str, optional
        Variable name (default is global variable vnm).
    
    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with new regional sums.
    """
    if df.empty:
        # If df is empty, initialize it with df_day
        df_updated = df_day.copy()
    else:
        df_updated = df.append(df_day)
    
    # save updated GFED regional sums to CSV
    fnm = dirData + 'Output/' + str(yr) + '/GFED5eNRTcumu' + vnm + '_' + str(yr) + '.csv'
    df_updated.to_csv(fnm)
         
    return df_updated

def readclimcumudata(vnm='EM'):
    pass
    return climcumudata

def generatecumufig(yr, vnm, df_updated=None):
    # If df_updated is not provided, read the updated cumulative data
    if df_updated is None:
        df_updated = readcumudata(yr, vnm='EM')

    # read climatological cumulative data
    climcumudata = readclimcumudata(vnm=vnm)

    # plot the updated data with climatological data
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    df_updated.plot(ax=ax, marker='o', title=f'Cumulative {vnm} Emissions for {yr}', grid=True)
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Cumulative {vnm} Emissions (gC)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # save the figure
    fig_path = dirData + 'Output/' + str(yr) + '/GFED5eNRTcumu' + vnm + '_' + str(yr) + '.png'
    fig.savefig(fig_path)
    
def update_cumufig(yr, mo, day, vnm='EM'):

    # load current cumulative emissions data
    df = readcumudata(yr, vnm=vnm)
    
    # read the daily data
    da = readGFED5eco(yr, mo=mo, day=day, vnm=vnm)
    
    # calculate global/regional sums for the daily data 
    df_day = doGFEDregsum(da, yr, mo, day)
    
    # update the cumulative emissions DataFrame
    df_updated = updateGFEDregsum(df, df_day, yr, vnm)
    
    # generate the updated cumulative figure
    generatecumufig(yr, vnm, df_updated=df_updated)
    

# Optional step : Convert GFED5eNRT from daily to monthly
def readGFED51NRT(yr,mo=None,day=None,vnm='EM'):
    """
    Read GFED5.1 NRT data for a given year, month, and day.
    """
    if day is not None:
        ds = xr.open_dataset(dirData + 'Output/'+vnm+'/'+str(yr)+'/'+vnm+'_'+str(yr)+'-'+str(mo).zfill(2)+'-'+str(day).zfill(2)+'.nc')
    elif mo is not None:
        ds = xr.open_mfdataset(dirData + 'Output/'+vnm+'/monthly/'+vnm+'_'+str(yr)+'-'+str(mo).zfill(2)+'*.nc')
    
    else:
        ds = xr.open_mfdataset(dirData + 'Output/'+vnm+'/'+str(yr)+'/'+vnm+'_'+str(yr)+'*.nc')
    return ds

def make_table_yr(yr,vnm='EM'):
    """
    Create a table of daily GFED5 NRT for a given year
    """

    # read GFED5 NRT data
    ds = readGFED51NRT(yr,vnm=vnm)
    da = ds[vnm].transpose('time', 'lct', 'lat', 'lon') # transpose to (time, lct, lat, lon)
    # read GFED mask
    daGFED = read_GFEDmask()

    # derive global sum dataframe 
    print(f'Deriving global sum for {yr}...')
    df = da.sum(dim=['lat','lon']).to_pandas()
    df['Total'] = df.sum(axis=1)

    df['Reg'] = 'Globe' # set multi-index (t, Reg='Globe')
    df.set_index([df.index,'Reg'], inplace=True)

    # derive regional sum dataframe and appended to df
    for iGFED in range(1,15):
        regnm = GFEDnms[iGFED]
        print(f'Deriving region {regnm} sum for {yr}...')
        maskreg = (daGFED==iGFED)  # region mask

        dfreg = (da*maskreg).sum(dim=['lat','lon']).to_pandas()
        dfreg['Total'] = dfreg.sum(axis=1)

        dfreg['Reg'] = regnm # set multi-index (t, Reg=reg)
        dfreg.set_index([dfreg.index,'Reg'], inplace=True)

        df = pd.concat([df,dfreg])

    # save df to csv file
    outfile = dirData + 'Output/'+vnm+'/'+vnm+'_'+str(yr)+'.csv'
    df.to_csv(outfile)

def read_table_yr(yr, vnm='EM'):
    """
    Read a table of GFED5 NRT for a given year.
    """
    import pandas as pd

    infile = dirData + 'Output/'+vnm+'/'+vnm+'_'+str(yr)+'.csv'
    df = pd.read_csv(infile)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index(['time','Reg']).sort_index(level='time')

    return df

def day_to_month(year,vnm='EM',missingdays=None):
    # read GFED5 NRT data
    ds = readGFED51NRT(year,vnm=vnm)
    da = ds[vnm].transpose('time', 'lct', 'lat', 'lon')

    # combine daily data to monthly data
    damon = da.groupby(da.time.dt.month).sum(dim='time')
    months = damon['month'].values
    time_values = [pd.Timestamp(f'{year}-{month:02d}-01') for month in months]
    damon = damon.assign_coords(time=('month', time_values))
    damon = damon.swap_dims({'month': 'time'})
    damon = damon.drop_vars('month')
    damon = damon.transpose('time', 'lct', 'lat', 'lon')

    # scale up by accounting for missing detections
    if missingdays is not None:
        for missingms, missingday in missingdays:
            missingdt = pd.Timestamp(f'{year}-{missingday:02d}-01')
            damon.loc[missingdt, :, :, :] = \
                damon.loc[missingdt, :, :, :] * (30 / (30 - missingday))

    # save to netcdf
    dsmon = xr.Dataset({vnm: damon})
    to_netcdf(dsmon, dirData + 'Output/'+vnm+'/monthly/'+vnm+str(year)+'.nc')

def make_monthlytable_yr(yr,vnm='EM'):
    """
    Create a table of monthly GFED5 NRT for a given year
    """
    # read GFED5 NRT data
    ds = xr.open_dataset(dirData + 'Output/'+vnm+'/monthly/'+vnm+str(yr)+'.nc')

    da = ds[vnm].transpose('time', 'lct', 'lat', 'lon') # transpose to (time, lct, lat, lon)
    # read GFED mask
    daGFED = read_GFEDmask()

    # derive global sum dataframe 
    print(f'Deriving global sum for {yr}...')
    df = da.sum(dim=['lat','lon']).to_pandas()
    df['Total'] = df.sum(axis=1)

    df['Reg'] = 'Globe' # set multi-index (t, Reg='Globe')
    df.set_index([df.index,'Reg'], inplace=True)

    # derive regional sum dataframe and appended to df
    for iGFED in range(1,15):
        regnm = GFEDnms[iGFED]
        print(f'Deriving region {regnm} sum for {yr}...')
        maskreg = (daGFED==iGFED)  # region mask

        dfreg = (da*maskreg).sum(dim=['lat','lon']).to_pandas()
        dfreg['Total'] = dfreg.sum(axis=1)

        dfreg['Reg'] = regnm # set multi-index (t, Reg=reg)
        dfreg.set_index([dfreg.index,'Reg'], inplace=True)

        df = pd.concat([df,dfreg])

    # save df to csv file
    outfile = dirData + 'Output/'+vnm+'/monthly/table_'+vnm+'_'+str(yr)+'.csv'
    df.to_csv(outfile)

# ----------------------------------------------------------------------------------------------------
# Bulk run functions
# ----------------------------------------------------------------------------------------------------

# NRT operational run functions
def run_1day(yr, mo, day, IMG=True, upd=False):
    """Generates GFED5eNRT data for a single day.

    This function converts Visible Infrared Imaging Radiometer Suite (VIIRS) Active Fire
    (VAF) data into Burned Area (BA) and Emissions (EM) using Emission Factor
    (EFA) and Fuel Consumption (FC) lookup tables.

    The VAF fire location data can be obtained from two sources:
    - Standard VNP14IMGML location data (when `IMG` is False).
    - Data downloaded and converted from VNP14IMG image data (when `IMG` is True).
      If VNP14IMG data is unavailable, VNP14IMG_NRT image data will be used.

    Args:
        yr (int): The year for which to generate data.
        mo (int): The month for which to generate data.
        day (int): The day for which to generate data.
        IMG (bool, optional): If True, download and convert VIIRS IMG data to IMGDL.
                               If False, use pre-downloaded VNP14IMGML data.
                               Defaults to True.
        upd (bool, optional): If True, forces an update of fire location data,
                              even if the files already exist. Defaults to False.
    """

    print(f"Running for {yr}-{mo}-{day}")

    print("...downloading data from NASA server...")
    if IMG:
        dlflag = make_VNP14IMGDL(yr, mo, day, upd=upd)
    else:
        print("...using pre-downloaded VNP14IMGML data...")
        dlflag = True

    if not dlflag:
        print(f"Warning: No data available for {yr}-{mo}-{day}.")
        return
    
    print("...reading and preprocessing VIIRS active fire location data...")
    dfFC = readpreprocess_DL(yr, mo, day, IMG=IMG)

    print("...recording VIIRS active fires in each 500m grid...")
    nVAFC = recordVAF500m(yr, mo, day, dfFC)

    print("...recording VIIRS active fires in each 0.25deg grid...")
    if nVAFC > 0:
        cal_VAFp25_alltiles_1day(yr, mo, day)
        print("...calculating scaled BA for each 0.25deg grid...")
        cal_BA_scled_day(yr, mo, day)
        print("...calculating scaled EM for each 0.25deg grid...")
        cal_EM_scled_day(yr, mo, day)

        print("...generating GFED5eNRTeco (16-class combined VAF, BA, EM) data...")
        make_GFED5eco(yr, mo, day)

        print("...generating GFED5eNRTspe (all-species EM data)...")
        make_EMspecies(yr, mo, day)
    else:
        print(f"No active fires were detected by VIIRS for {yr}-{mo}-{day}")

def get_GFED5_lastday(yr):
    """Determines the latest date for which GFED5eNRT data has been created.

    Args:
        yr (int): The year for which to check the latest date.

    Returns:
        datetime.date or None: The latest date as a datetime.date object if files are found,
                               otherwise None.
    """
    import os
    from datetime import datetime

    # Construct the directory path for the specified data type
    dirGFED = os.path.join(dirData, 'Output', str(yr))

    # List all files in the directory
    try:
        files = os.listdir(dirGFED)
    except FileNotFoundError:
        print(f"Warning: Directory not found: {dirGFED}")
        return None

    if not files:
        return None  # No matching files found

    # Extract dates from file names.
    dates = []
    for file in files:
        try:
            dates.append(datetime.strptime(file[-13:-3], '%Y-%m-%d').date())
        except ValueError:
            # Skip files that don't match the expected naming convention
            continue

    if not dates:
        return None # No valid dates extracted from filenames

    # Find and return the latest date
    latest_date = max(dates)
    return latest_date

def update_to_now():
    """Wrapper function to update GFED5eNRT data to the latest available VIIRS data.

    This function checks the last date for which GFED5eNRT data exists. If the
    current date is newer than the last available GFED5eNRT data, it iteratively
    calls `run_1day` for each missing day to update the dataset.
    """
    from datetime import datetime, timedelta

    # Get the current date and the last available date of GFED5eNRT EM data
    date_now = datetime.now().date()
    date_GFED5 = get_GFED5_lastday(date_now.year)

    # If no GFED5 data is found, start updating from the beginning of the current year
    if date_GFED5 is None:
        date_GFED5 = datetime(date_now.year, 1, 1).date()
        print(f"No existing GFED5 data found. Starting update from {date_GFED5}.")
    else:
        print(f"Last GFED5 data available up to: {date_GFED5}")
    
    # If the last availabe date is before the current date, call `run_1day()` for each day between the two dates
    if date_now > date_GFED5:
        print(f"VIIRS data is newer than GFED5 data, updating to {date_now}")
        date_difference = date_now - date_GFED5

        # Iterate through each day that needs to be updated
        for day_offset in range(1, date_difference.days + 1):
            date_run = date_GFED5 + timedelta(days=day_offset)
            print(f"Processing update for {date_run.year}-{date_run.month}-{date_run.day}...")
            # Always use the IMG option for operational update
            run_1day(date_run.year, date_run.month, date_run.day, IMG=True)
    else:
        print("GFED5 data is already up to date with VIIRS data.")

# other bulk run functions
def run_1mon(yr, mo, IMG=False, upd=False):
    ''' generate GFED5eNRT for all days in a single month: call `run_1day()` for each day
    '''
    import calendar
    from tqdm import trange

    ndays = calendar.monthrange(yr, mo)[1]
    for day in trange(1, ndays+1):
        run_1day(yr, mo, day, IMG=IMG, upd=upd)

# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # nowarn()

    # ------------------------------------------------------------------
    # GFED5eNRT run
    # -------------

    update_to_now()  # Update GFED5eNRT to the latest date with VIIRS data

    # run_1day(2025,7,20, IMG=True, upd=True)  # Derive GFED5 betaV BA and EM for 1 day
    # run_1day(2025,1,1, IMG=True, upd=False)
    # run_1mon(2025,3,IMG=True, upd=True)

    # ------------------------------------------------------------------
    # GFED5e run
    # -----------

    # run_1mon(2024,1,IMG=False)

    # ------------------------------------------------------------------

