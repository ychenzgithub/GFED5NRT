""" GFED5e NRT
   ----------------

    This is a stand-alone python code to run the GFED5 extension NRT version (GFED5NRT) burned area (BA) and emissions (EM).
   
    In this code, we derived the extension version of daily NRT GFED5 burned area and emissions (at 0.25 deg resolution) from the VIIRS 375m active fire data. The VIIRS active fire counts are recorded and categorized to different burning types based on land cover types. The GFED5NRT burned area and emissions are calculated based on 2-step scaling approaches. First, pre-derived effective fire area (EFA) scalars were used to convert the daily VIIRS active fire number (VAF) to daily burned area at 0.25 degree resolution (GFED5NRT BA). Then, pre-calculated fuel consumption (FC) scalars were combined with GFED5NRT BA to derive daily burned area at 0.25 degree resolution (GFED5NRT EM). We combined the output data into two data streams: GFED5NRTeco contains VAF, BA and EM data for 16 classes; GFED5NRTspe contains EM data for individual chemical (gas and aerosol) species.

    See `README.md` for more detail about this code.
    
    Yang Chen (yang.chen@uci.edu)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import userconfig
import pyproj
import datetime
import concurrent.futures
import time
from typing import Tuple, List, Any
import calendar
import warnings
warnings.filterwarnings(
    action="ignore", 
    category=FutureWarning,
    module="osgeo.gdal"
)

dirData = userconfig.dirData  # project directory

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
         'Barren_B', 'Boreal_FS', 'Tundra_T', 'Boreal_C']        
LCnms_full = ['Water','Forest boreal','Forest tropical','Forest temperate','Sparse temperate mosaic','Shrublands tropical','Shrublands temperate','Grasslands temperate','Savanna woody','Savanna open','Grasslands tropical','Wetlands','Croplands tropical','Urban','Croplands temperate','Snow/ice','Barren','Sparse boreal forest','Tundra','Croplands boreal']

# 16-class Land cover names for GFED5.1 emissions
EMLCnms = ['Tundra','Sparse boreal forest','Boreal forest','Temperate grassland','Temperate shrubland','Temperate mosaic','Temperate forest','Tropical grassland','Tropical shrubland','Open savanna','Woody savanna','Tropical forest','Other','Cropland','Peat','Deforestation']

# --- MODIS CONSTANTS ---
T = 1111950.5197665233
Xmin = -20015109.355797417
Ymax = 10007554.677898709
R0 = 6371007.181000       # Earth radius in [m]
LIMIT_LEFT = -20015109.354 # left limit of MODIS grid in [m]
LIMIT_TOP = 10007554.677  # top limit of MODIS grid in [m]
TILE_SIZE_METERS = R0 * np.pi / 18. # m, height/width of MODIS tile
NDIM_500M = 2400         # Number of cells (2400)
REAL_RES_500M = ((abs(LIMIT_LEFT) * 2) / 36) / NDIM_500M # actual size for each pixel (500m)

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
    doy = datetime.datetime(yr, mo, day).timetuple().tm_yday
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

def sftp_upload(
    ftpurl,
    ftpun,
    ftppw,
    local_path,
    remote_dir="/",
    ftpport=22
):
    """
    Upload a file to an SFTP server.

    Parameters
    ----------
    ftpurl : str
        The SFTP server URL or IP address.
    local_path : str
        The path of the local file to upload.
    remote_dir : str, optional
        The remote directory on the SFTP server where the file will be uploaded; default is '/'.
    ftpport : int, optional
        The port number for the SFTP connection; default is 22.

    Returns
    -------
    None
    """
    import paramiko
    
    # Retrieve credentials from environment variables for security
    # ftpun = os.environ.get("SFTP_UN")
    # ftppw = os.environ.get("SFTP_PW")

    # Exit if credentials aren't found
    if not ftpun or not ftppw:
        print("Error: SFTP username or password not found in environment variables.")
        return

    # Set remote destination path
    local_filename = os.path.basename(local_path)
    remote_path = os.path.join(remote_dir, local_filename).replace("\\", "/") # Handles Windows paths
    
    transport = None
    sftp = None
    try:
        # Establish the SFTP connection
        transport = paramiko.Transport((ftpurl, ftpport))
        transport.connect(username=ftpun, password=ftppw)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Upload the file
        sftp.put(local_path, remote_path)
        
        print(f"File '{local_filename}' successfully uploaded to '{ftpurl}:{remote_path}'")
        
    except Exception as e:
        print(f"Error occurred during SFTP upload: {e}")
        
    finally:
        # Close the SFTP client and transport
        if sftp:
            sftp.close()
        if transport:
            transport.close()

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

def upload_file_WUR(fnmout):
    # upload data to WUR server
    try:
        sftp_upload(userconfig.ftpurl_WUR, userconfig.ftpun_WUR, userconfig.ftppw_WUR,
            fnmout, remote_dir=userconfig.ftpdir_WUR, ftpport=userconfig.ftpport_WUR)
    except:
        print(f'The uploading of {fnmout} to WUR ftp server is unsuccessful')

# functions used to read intermediate and output data
def read_BA(yr,mo,day,sat='VNP'):
    """ read BA dataset from the output directory
    """
    import xarray as xr
    import os
    fnm = dirData+'Intermediate/BA/'+str(yr)+'/BA_'+sat+'_'+strymd(yr,mo,day)+'.nc'
    if os.path.exists(fnm):
        ds = xr.open_dataset(fnm)
    else:
        ds = None
    return ds

def read_EM(yr,mo,day,sat='VNP'):
    """ read EM dataset from the output directory
    """
    import xarray as xr
    ds = xr.open_dataset(dirData+'Intermediate/EM/'+str(yr)+'/EM_'+sat+'_'+strymd(yr,mo,day)+'.nc')
    return ds

def read_VNP14IMG_NRT_daily(yr,mo,day,sat='VNP'):
    import pandas as pd
    stryr = str(yr)
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
    strdoy = str(doy).zfill(3)
    fnm = dirData + 'Input/'+sat+'14IMGDL/'+sat+'14IMGDL_'+stryr+strdoy+'.csv'
    df = pd.read_csv(fnm,index_col=0)
    return df

def read_VNP14IMGML(yr,mo,day=None,sat='VNP',ver='C2.04', usecols=["YYYYMMDD", "HHMM", "Lat", "Lon", "Line", "Sample", "FRP", "Confidence", "Type", "DNFlag"]):
    """ Optionally use VIIRS monthly or daily data from monthly standard data (VNP14IMGML or VJ114IMGML)
    """       
    # set monthly VNP14IMGML or VJ114IMGML file name
    if sat == 'VNP':
        dirin = os.path.join(dirData, 'Input', 'VNP14IMGML')
    elif sat == 'VJ1':
        dirin = os.path.join(dirData, 'Input', 'VJ114IMGML')
    
    fnmFC = os.path.join(dirin,sat+'14IMGML.' + str(yr) + str(mo).zfill(2) + '.' + ver + '.csv')
    
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
        if day is not None:
            df = df[df.YYYYMMDD == pd.to_datetime(str(yr)+'-'+str(mo).zfill(2)+'-'+str(day).zfill(2))]

        return df
    else:
        print('No data available for file',fnmFC)
        return None
    
def read_VNP14IMGML_daily(yr,mo,day,ver='C2.02', usecols=["YYYYMMDD", "HHMM", "Lat", "Lon", "Line", "Sample", "FRP", "Confidence", "Type", "DNFlag"]):
    """ 
    !!! Should be removed !!!
    
    Optionally use VIIRS daily data from monthly standard data (VNP14IMGML)
    """    
    dirVNP14IMGML = os.path.join(dirData, 'Input', 'VNP14IMGML')
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

def read_GFED5eco(yr,mo,day,sat='VNP'):
    """ read daily GFED5eco dataset from the output directory
    """
    import xarray as xr
    import os

    fnm = dirData+'Output/'+str(yr)+'/GFED5NRTeco_'+sat+'_'+strymd(yr,mo,day)+'.nc'
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

class MODISTileProcessor:
    """
    A class to hold constants and pre-initialized pyproj transformer,
    optimized for fast MODIS tile geotransform and bounding box calculation.
    """
    def __init__(self):
        # 1. Pre-calculate grid cell size
        ng = NDIM_500M
        self.w = T / ng

        # 2. Pre-create i (col) and j (row) grids (constant for 2400x2400)
        self.i_grid, self.j_grid = np.meshgrid(np.arange(NDIM_500M), np.arange(NDIM_500M))
        
        # 3. Pre-initialize the constant pyproj Transformer (CRITICAL SPEEDUP)
        sinus_prj4 = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
        self.transformer_inverse = pyproj.Transformer.from_proj(
            pyproj.Proj(sinus_prj4),    # from MODIS Sinusoidal
            pyproj.Proj("EPSG:4326"),   # to WGS84 (lon/lat)
            always_xy=True
        )

        # 4. Pre-select boundary indices for fast Bounding Box calculation
        n = NDIM_500M
        
        # Get the i, j indices for the border of the tile (top, bottom, left, right edges)
        border_i = np.concatenate([
            self.i_grid[0, :],         # Top edge (j=0)
            self.i_grid[-1, :],        # Bottom edge (j=n-1)
            self.i_grid[:, 0],         # Left edge (i=0)
            self.i_grid[:, -1]         # Right edge (i=n-1)
        ])
        border_j = np.concatenate([
            self.j_grid[0, :],         # Top edge (j=0)
            self.j_grid[-1, :],        # Bottom edge (j=n-1)
            self.j_grid[:, 0],         # Left edge (i=0)
            self.j_grid[:, -1]         # Right edge (i=n-1)
        ])
        
        # Use only unique indices (to avoid transforming the 4 corners multiple times)
        unique_indices = np.unique(np.stack([border_i, border_j], axis=1), axis=0)
        self.border_i_flat = unique_indices[:, 0]
        self.border_j_flat = unique_indices[:, 1]
        
    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def MODtileij2xy_array(self, H, V):
        """Vectorized computation of x, y coordinates for the full grid."""
        # Compute x and y for the entire grid
        x = (self.i_grid + 0.5) * self.w + H * T + Xmin
        y = Ymax - (self.j_grid + 0.5) * self.w - V * T
        return x, y

    def sinusproj_array_optimized(self, x, y):
        """Optimized sinusoidal projection using the pre-initialized transformer."""
        # This is fast because it's only called on the boundary points now
        lon, lat = self.transformer_inverse.transform(x, y)
        return lon, lat

    def MODtilegt_optimized(self, H, V):
        """Get the geotransform information in a MODIS tile (H,V)."""
        
        # define geotransformation
        gt0 = LIMIT_LEFT + H * TILE_SIZE_METERS
        gt3 = LIMIT_TOP - V * TILE_SIZE_METERS
        gt1 = REAL_RES_500M
        gt5 = -REAL_RES_500M
        gt = [gt0, gt1, 0, gt3, 0, gt5]
        return gt

    def getMODlatlon_boundary(self, vh, vv):
        """
        Calculates lon/lat only for the boundary coordinates of the tile.
        This is done to quickly derive the bounding box.
        """
        # 1. Get x, y coordinates for ALL cells
        x_all, y_all = self.MODtileij2xy_array(vh, vv)
        
        # 2. Extract only the boundary coordinates using pre-calculated indices
        x_border = x_all[self.border_j_flat, self.border_i_flat]
        y_border = y_all[self.border_j_flat, self.border_i_flat]
        
        # 3. Convert ONLY the boundary x, y to lon, lat
        lon_border, lat_border = self.sinusproj_array_optimized(x_border, y_border)
        
        return lon_border, lat_border

    def getMODlatlon(self, vh, vv):
            """Read lat/lon for a MOD tile using pre-initialized objects."""
            # Get x, y coordinates for the MOD tile
            x, y = self.MODtileij2xy_array(vh, vv)
            
            # Convert x, y to lon, lat using sinusoidal projection
            # This call is now much faster!
            lon, lat = self.sinusproj_array_optimized(x, y)
            
            return lon, lat

    # --------------------------------------------------------------------------
    # Target Function
    # --------------------------------------------------------------------------

    def get_tile_paras(self, vh, vv):
        ''' return parameters of a tile
        '''
        strhv = 'h' + str(vh).zfill(2) + 'v' + str(vv).zfill(2)
        xs, ys = NDIM_500M, NDIM_500M
        
        # 1. Get Geotransform (Fast)
        MODgt = self.MODtilegt_optimized(vh, vv)
        
        # 2. Get Boundary Lat/Lon (Optimized and Fast)
        Tlon_border, Tlat_border = self.getMODlatlon_boundary(vh, vv)
        
        # 3. Calculate Bounding Box (Fast)
        # Bounding box (min_lat, max_lat, min_lon, max_lon)
        bb = (
            np.nanmin(Tlat_border), 
            np.nanmax(Tlat_border), 
            np.nanmin(Tlon_border), 
            np.nanmax(Tlon_border)
        )

        # Note: If you need the full Tlon and Tlat arrays later, you will need to 
        # run the full (slow) transformation separately, but for the parameters
        # requested, this is complete and highly optimized.

        return strhv, xs, ys, MODgt, bb

def set_FCtile_ds(arrFC_month):
    ''' Based on the input 2d array of FC data, generate a dataset that is formatted the same as MCD64A1 (using a sample MCD64A1 hdf file as reference)
    '''
    import xarray as xr
    
    # read any 500m MCD64A1 hdf file (4800x4800)
    fnmhdf = dirData+'Input/Sample.MCD64A1.hdf'  
    # dsMOD = xr.open_dataset(fnmhdf,engine='pynio')
    # dsMOD = dsMOD.assign(FC = dsMOD['Burn_Date'])

    dsMOD = xr.open_dataset(fnmhdf,engine='netcdf4', decode_timedelta=True)
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
# GFEDextNRT steps
# ----------------------------------------------------------------------------------------------------

# Step 0: VIIRS data download and pre-processing
def read_earthdata_token():
    """ Extract saved Earthdata token from system
    """
    token = os.environ.get("EARTHDATA_PAT")
    if token is None:
        print(f"❌ Error: Please set your EARTHDATA_PAT environment variable")
        return None        
    else:
        return token
    
# the following old approach is not safe, and should be removed (tother with the Earthdata_token.txt)
def read_earthdata_token_old():
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

def geturl_VNP14IMG_daily(yr,mo,day,NRT=False,sat='VNP'):
    """ return the remote VNP14IMG daily data path url"""

    # data source from MODAPS and LADS
    modaps_svr = "https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/"
    lads_svr = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/"

    # pick source based on NRT flag
    if NRT:
        base_url = modaps_svr + f'{sat}14IMG_NRT/'
    else:
        base_url = lads_svr + f'{sat}14IMG/'

    # for NOAA21(J2), only data source is MODAPS
    if sat == 'VJ2':
        base_url = modaps_svr + f'{sat}14IMG_NRT/'

    # the final daily data path
    url = base_url + str(yr) + '/' + strdoy(yr,mo,day) + '/'
    return url

def checkts_VNP14IMG_daily(edl_token, yr,mo,day, NRT=False, sat='VNP'):
    """ check if VNP14IMG for all times in a day is available"""

    # if NRT:
    #     base_url="https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG_NRT/"
    # else:
    #     base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG/"
    # url = base_url + str(yr) + '/' + strdoy(yr,mo,day) + '/'

    url = geturl_VNP14IMG_daily(yr,mo,day,NRT=NRT,sat=sat)
    # download the data only if the daily data is complete
    available_times = get_remote_ts(url, edl_token)
    if len(available_times) == 0:
        return False 
    if max(available_times) == '2354':
        return True
    else:
        return False

def checkempty_VNP14IMG_daily(edl_token, yr,mo,day, NRT=False, sat='VNP'):
    """ check if VNP14IMG in a day is empty"""

    # if NRT:
    #     base_url="https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG_NRT/"
    # else:
    #     base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG/"
    # url = base_url + str(yr) + '/' + strdoy(yr,mo,day) + '/'

    url = geturl_VNP14IMG_daily(yr,mo,day,NRT=NRT,sat=sat)
    # download the data only if the daily data is complete
    available_times = get_remote_ts(url, edl_token)
    if len(available_times) == 0:
        return True 
    else:
        return False

def download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=False, sat='VNP'):
    """
    Downloads Suomi-NPP VIIRS active fire data from LANCE using wget with authentication and specific parameters.

    Args:
        base_url (str): The URL of the VIIRS active fire data to download.
        edl_token (str): The Earthdata Login token for authentication.
    """

    # if NRT:
    #     base_url="https://nrt3.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG_NRT/"
    # else:
    #     base_url="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/VNP14IMG/"
    # url = base_url + str(yr) + '/' + strdoy(yr,mo,day) + '/'
    
    url = geturl_VNP14IMG_daily(yr,mo,day,NRT=NRT,sat=sat)
    py_wget(url, edl_token, dirData+'Input/', no_if_modified_since=no_if_modified_since)

def check_VNP14IMG_presence(yr,mo,day,sat='VNP'):
    """ check if local VNP14IMG data is available for a day """
    from glob import glob

    stryr = str(yr)
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
    strdoy = str(doy).zfill(3)

    fnms = glob(dirData + 'Input/'+sat+'14IMG/'+stryr+'/'+strdoy+'/'+sat+'14IMG.A'+stryr+strdoy+'*.nc')

    filepresenceflag = (len(fnms) > 0)

    return filepresenceflag

def delete_subdirs_and_files(directory_path):
    """
    Deletes all subdirectories and files within the specified directory.

    Args:
        directory_path (str): Path to the directory to clean.
    """

    import shutil
    if not os.path.isdir(directory_path):
        print(f"Not a directory: {directory_path}")
        return

    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        try:
            if os.path.isfile(full_path) or os.path.islink(full_path):
                os.remove(full_path)
                print(f"Deleted file: {full_path}")
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(f"Deleted directory: {full_path}")
        except FileNotFoundError:
            print(f"Not found: {full_path}")
        except PermissionError:
            print(f"Permission denied: {full_path}")
        except Exception as e:
            print(f"Error deleting {full_path}: {e}")

def convert_VNP14IMG_to_DL(yr,mo,day,clean=False,sat='VNP'):
    """
    This function reads VNP14 netCDF files (either standard or NRT data) and extracts fire pixel data (text) into fire location data (VNP14IMGDL).
    """
    from glob import glob
    import xarray as xr 
    import pandas as pd
    stryr = str(yr)
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday

    # get VNP14IMG file names for the day (if not available, use VNP14IMG_NRT files)
    if check_VNP14IMG_presence(yr,mo,day,sat=sat):
        fnms = glob(dirData + 'Input/'+sat+'14IMG/'+stryr+'/'+strdoy(yr,mo,day)+'/'+sat+'14IMG.A'+stryr+strdoy(yr,mo,day)+'*.nc')
    else:
        print('No VNP14IMG data found for '+stryr+strdoy(yr,mo,day)+', use VNP14IMG_NRT instead')  
        fnms = glob(dirData + 'Input/'+sat+'14IMG_NRT/'+stryr+'/'+strdoy(yr,mo,day)+'/'+sat+'14IMG_NRT.A'+stryr+strdoy(yr,mo,day)+'*.nc')

    # convert image data to daily fire location (DL) data 
    dfFP = pd.DataFrame(columns=['Lon','Lat','FRP','Sample','Confidence','DNFlag'])
    for fnm in fnms:
        ds = xr.open_dataset(fnm)
        if ds.attrs['FirePix']>0:
            sFP = pd.DataFrame({'Lon':ds.FP_longitude.values,'Lat':ds.FP_latitude.values,'FRP':ds.FP_power.values,'Sample':ds.FP_sample.values,'Confidence':ds.FP_confidence.values,'DNFlag':ds.FP_day.values})
            dfFP = pd.concat([dfFP,sFP],axis=0)
    dfFP = dfFP.reset_index(drop=True)

    # save DL data to VNP14IMGDL
    dirout = dirData+'Input/'+sat+'14IMGDL'
    mkdir(dirout)
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
    if len(dfFP) > 0:
        dfFP.to_csv(dirout+'/'+sat+'14IMGDL_'+str(yr)+str(doy).zfill(3)+'.csv')

    # delete VNP14IMG data to save space
    if clean:
        delete_files(fnms)

    return dfFP

def make_VNP14IMGDL(yr,mo,day,upd=False,sat='VNP',comp=True):
    """ for a day, download VNP14IMG, VJ114IMG, VJ214IMG netcdf and convert to location (DL) data"""
    import os

    # if VNP14IMGDL data already available and upd is not set to True, skip the creation
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
    fileext = os.path.exists(dirData+'Input/'+sat+'14IMGDL/'+sat+'14IMGDL_'+str(yr)+str(doy).zfill(3)+'.csv')
    if fileext & (not upd):
        print(f'...{sat}14IMGDL data already exists...')
        return True

    # otherwise, download VNPIMG or VNPIMG_NRT data and convert to fire location data
    edl_token = read_earthdata_token() # get the earthdata token
    dlflag = False
    
    if comp:  # only download when the last time step in a day is 2354
        VNP14IMG_data_complete = checkts_VNP14IMG_daily(edl_token, yr,mo,day, NRT=False, sat=sat)
        VNP14IMG_NRT_complete = checkts_VNP14IMG_daily(edl_token, yr, mo, day, NRT=True, sat=sat)
        if VNP14IMG_data_complete: # preferred data source is VNP14IMG in LAADS
            print("...downloading VNP14IMG data...")
            dlflag = download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=False, sat=sat)  # force redownload by `no_if_modified_since`
            print("...converting VIIRS data...")
            convert_VNP14IMG_to_DL(yr, mo, day, clean=False, sat=sat) 
            dlflag = True
        elif VNP14IMG_NRT_complete:  # only if VNP14IMG not available, try downloading VNP14IMG_NRT data from LANCE
            print("...downloading VNP14IMG_NRT data...")
            dlflag = download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=True, sat=sat)
            print("...converting VIIRS data...")
            convert_VNP14IMG_to_DL(yr, mo, day, clean=False, sat=sat) 
            dlflag = True
    else:  # even the last timestep is not 2354, download the files
        VNP14IMG_data_empty = checkempty_VNP14IMG_daily(edl_token, yr,mo,day, NRT=False, sat=sat)
        VNP14IMG_NRT_empty = checkempty_VNP14IMG_daily(edl_token, yr, mo, day, NRT=True, sat=sat)
        if not VNP14IMG_data_empty: # preferred data source is VNP14IMG in LAADS
            print("...downloading VNP14IMG data...")
            dlflag = download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=False, sat=sat)  # force redownload by `no_if_modified_since`
            print("...converting VIIRS data...")
            convert_VNP14IMG_to_DL(yr, mo, day, clean=False, sat=sat) 
            dlflag = True
        elif not VNP14IMG_NRT_empty:  # only if VNP14IMG not available, try downloading VNP14IMG_NRT data from LANCE
            print("...downloading VNP14IMG_NRT data...")
            dlflag = download_VNP14IMG_daily(edl_token, yr, mo, day, no_if_modified_since=True, NRT=True, sat=sat)
            print("...converting VIIRS data...")
            convert_VNP14IMG_to_DL(yr, mo, day, clean=False, sat=sat) 
            dlflag = True        
    return dlflag


# Step 1: Record VAF at 500m resolution (MODIS 500m sinusoidal grids)
def readpreprocess_DL(yr,mo,day,IMG=True,sat='VNP'):
    ''' read and preprocess the VIIRS daily active fire location data
    '''
    # read VIIRS active fire data
    if IMG: 
        dfFC = read_VNP14IMG_NRT_daily(yr,mo,day,sat=sat)
        dfFC = removestaticpixels(dfFC) # filter out static pixels 
    else:
        dfFC = read_VNP14IMGML(yr, mo, day=day)

    # add major LCT
    dfFC = adddfmjLCT(dfFC)

    # add DN flag
    dfFC = addDN(dfFC, NRT=IMG)

    # add sample adjustment weight
    dfFC = add_wgt_2_VNP(dfFC, mo)

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
    if len(df) == 0:
        return df

    # get dfwgt for the month
    dfwgt = pd.read_csv(dirData+'Input/VIIRSsplwgt_2019-2021.csv',index_col=0)
    dfwgtmidx = dfwgt.reset_index().set_index(['Sample', 'mo','DN','btype'])
    dfwgtmidxmo = dfwgtmidx.xs(mo,level='mo')

    # Use pandas' `.join` method to match and fetch 'wgt' values
    df = df.set_index(['Sample', 'DN', 'btype'])  # Temporarily set the same index as 'dfwgt'
    df['wgt'] = dfwgtmidxmo['wgt']  # Fetch 'wgt' values based on the MultiIndex
    df = df.reset_index()  # Reset index back to default

    return df

def save_VAF_1tile_day(dsFC,yr,mo,day,strhv,sat='VNP'):
    ''' save daily FC (create directory if necessary)
    '''
    import os

    dirFC = dirData+'Intermediate/'+sat+'500m/'+str(yr)

    # if directory not there, create one
    if not os.path.exists(dirFC):
        os.makedirs(dirFC)

    # save data to file
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
    fnmFC = dirFC+'/FC_'+str(doy).zfill(3)+'_'+strhv+'.nc'    
    to_netcdf(dsFC,fnmFC)

def check_VAF_1tile_day(yr,mo,day,strhv,sat='VNP'):
    ''' check if daily FC data file exists
    '''
    import os

    dirFC = dirData+'Intermediate/'+sat+'500m/'+str(yr)

    # if directory not there, create one
    if not os.path.exists(dirFC):
        os.makedirs(dirFC)

    # save data to file
    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
    fnmFC = dirFC+'/FC_'+str(doy).zfill(3)+'_'+strhv+'.nc'    

    return os.path.exists(fnmFC)

def cal_VAF_1tile_1day(dfFC,vh,vv,yr,mo,day,sat='VNP',processor=None):
    ''' record and save 1 tile 1day VAF data to MODIS-500m grid cells
    '''    
    # define parameters of the tile
    if processor is None:
        strhv,xs,ys,MODgt,bb = get_tile_paras(vh,vv)
    else:
        strhv, xs, ys, MODgt, bb = processor.get_tile_paras(vh, vv)
        
    # filter VIIRS active fire data using the bounding box of the tile
    dfFC_sub = filter_VNP14IMGML_NRT(dfFC,latlon=bb)

    if len(dfFC_sub) > 0:
        # convert pixel location to VAF numbers in each 500m grid cells 
        arrFC = FCpoints2arr(dfFC_sub,MODgt,xs,ys,strlon='Lon',strlat='Lat',sumcol='wgt')

        # only for tiles with nonzero active fire detections
        if arrFC.sum() > 0:
            # convert numerical dataarray to FC dataset
            dsFC = set_FCtile_ds(arrFC)

            # save FC (create directory if necessary)
            save_VAF_1tile_day(dsFC,yr,mo,day,strhv,sat=sat)

def recordVAF500m(dfFC, yr,mo,day=None, vhs=[0,35],vvs=[0,17],sat='VNP', processor = None):
    ''' wrapper to record global (all tiles, single day or month) VIIRS (C1) active fire (VAF) data to MODIS-500m resolution
    '''
    # loop over all tiles and record VAF data
    if len(dfFC) > 0:
        # # add sample adjustment weight
        # dfFC = add_wgt_2_VNP(dfFC, mo)

        for vh in range(vhs[0],vhs[1]+1):
            for vv in range(vvs[0],vvs[1]+1):
                # print(f'Processing tile h{vh}v{vv}...')
                strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)
                if not check_VAF_1tile_day(yr,mo,day,strhv,sat=sat):  # if file exist, skip
                    cal_VAF_1tile_1day(dfFC,vh,vv,yr,mo,day,sat=sat, processor = processor)
    else:
        if day is None:
            print('No data available for ', yr, mo)
        else:
            print('No data available for ', yr, mo, day)
    
    return len(dfFC)

# Step 2: Record VAF sums (GBA groups format) at 0.25deg resolution
def getVAFnc_day(yr,mo,day,vh,vv,sat='VNP'):
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


    strhv = 'h'+str(vh).zfill(2)+'v'+str(vv).zfill(2)

    dirFC = dirData+'Intermediate/'+sat+'500m/'+str(yr) 

    doy = datetime.datetime(yr,mo,day).timetuple().tm_yday
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
        
def cal_VAFp25_1tile_1day(yr,mo,day,vh,vv,sat='VNP'):
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
    FC = getVAFnc_day(yr,mo,day,vh,vv,sat=sat)
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

def cal_VAFp25_alltiles_1day(yr,mo,day,vhs=[0,35],vvs=[0,17],sat='VNP'):
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
            area_1tile = cal_VAFp25_1tile_1day(yr,mo,day,vh,vv,sat=sat)
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

    return ds_VAF_types, ds_VAF_bioms

def sav_VAFp25_alltiles_1day(ds_VAF_types,ds_VAF_bioms,yr,mo,day,sat='VNP'):
    # save to netcdf
    dirout = dirData+'Intermediate/'+sat+'AF/'+str(yr)
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

def read_VAFp25_alltiles_1day(yr, mo, day, sat='VNP'):
    """ read daily VIIRS AF data at 0.25 deg
    """
    import xarray as xr 

    fnmVAF = dirData+'Intermediate/'+sat+'AF/'+str(yr)+'/VAF_'+strymd(yr,mo,day)+'.nc'
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

def cal_BA_scled_day(yr, mo, day,sat = 'VNP'):
    """ Use EFA scalars to convet daily VIIRs active fire counts to BA; convert to G16 format and save to netcdf
    """
    import xarray as xr
    import numpy as np

    # read GFED mask
    daGFED = read_GFEDmask()

    # read daily VIIRS AF data (0.25 deg, GBA format)
    dsAF_yr, dsbAF_yr = read_VAFp25_alltiles_1day(yr, mo, day, sat=sat)

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
    fnmout = dirout+'/BA_'+sat+'_'+strymd(yr,mo,day)+'.nc'
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

def cal_EM_scled_day(yr, mo, day, sat='VNP'):
    """ Use FC scalars to convet daily VIIRS BA to EM; keep G16 format and save to netcdf
    """
    import xarray as xr 
    import numpy as np

    # read annual GFED5.1 original BA (km2) or GFED5.1 BAfromVAF (km2)
    dsBA_16class = read_BA(yr, mo, day, sat=sat)
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
    fnmout = dirout+'/EM_'+sat+'_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_16class_yr, fnmout)

# Step 5: Derive GFED5NRTeco (16-class combined data VAF + BA + EM)
def add_GFED5eco_attrs(ds):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ds.attrs['long_name'] = 'Global Fire Emissions Database v5eNRT ecosystem'
    ds.attrs['standard_name'] = 'GFED5NRTeco'
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

def getVAF16class(yr,mo,day,sat='VNP'):
    """ convert the original GFED5.1 BA dataset (biom + type, in km2) to the GFED5.1 16-class dataarray (in m2)

    - The input dataset `dsp25` is read from dsp25 = xr.open_mfdataset(dirGFED51BA+'BA'+str(yr)+'??.nc').load()
    - damaskp25TropFor, damaskp25TempFor, damaskp25Other are the 0.25 deg LCT fractions
    - FTC0103 is the forest fraction averaged over 2001-2003, read from readFTC0103()
    """
    import xarray as xr

    # read daily VIIRS AF data (0.25 deg, GBA-2groups format)
    dsAF_yr, dsbAF_yr = read_VAFp25_alltiles_1day(yr, mo, day, sat=sat)

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

def make_GFED5eco(yr,mo,day, sat='VNP'):
    """ createt GFED5e ecosystem 16-class data, which contains three layers
     - VAF: VIIRS active fire counts
     - BA: burned area
     - EM: emissions
    """
    # read 16-class BA data as the base
    ds_combined = read_BA(yr, mo, day, sat=sat)

    # add global attributes
    ds_combined = add_GFED5eco_attrs(ds_combined)

    # add BA attributes
    ds_combined = add_BA_attrs(ds_combined)

    # read 16-class EM data and add to ds_combined
    dsEM_16class = read_EM(yr, mo, day, sat=sat)
    ds_combined['EM'] = dsEM_16class['EM']

    # add EM attributes
    ds_combined = add_EM_attrs(ds_combined)

    # read GBA VAF data
    daVAF_16class = getVAF16class(yr,mo,day,sat=sat)
    ds_combined['VAF'] = daVAF_16class

    # add VAF attributes
    ds_combined = add_VAF_attrs(ds_combined)

    # save output
    dirout = dirData+'Output/'+str(yr)
    mkdir(dirout)
    fnmout = dirout+'/GFED5NRTeco_'+sat+'_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_combined, fnmout)
    
    return fnmout 

# Step 6: Derive GFED5NRTspe (species emissions
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ds.attrs['long_name'] = 'Global Fire Emissions Database v5eNRT species emissions'
    ds.attrs['standard_name'] = 'GFED5NRTspe'
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

def make_EMspecies(yr,mo,day,sat='VNP'):
    """ save 6-class EM files
    """

    # read emission data from GFED5NRT ecosystem
    EM16 = read_GFED5eco(yr,mo,day,sat=sat).EM 

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
    fnmout = dirout+'/GFED5NRTspe_'+sat+'_'+strymd(yr,mo,day)+'.nc'
    to_netcdf(ds_EM6_species_sum, fnmout)

    return fnmout


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
# GFEDextNRT visualization functions
# ----------------------------------------------------------------------------------------------------

def readGFED5eco(yr, mo=None, day=None, vnm='EM', sat=None):
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

    if sat is None:
        fnmhead = dirData + 'Output/' + str(yr) + '/GFED5NRTeco_'
    else:
        fnmhead = dirData + 'Output/' + str(yr) + '/GFED5NRTeco_' + sat + '_'

    if day is not None:
        fnm = fnmhead + str(yr) + '-' + str(mo).zfill(2) + '-' + str(day).zfill(2) + '.nc'
        if os.path.exists(fnm):
            ds = xr.open_dataset(fnm)
        else:
            return None
    elif mo is not None:
        fnms = glob(fnmhead + str(yr) + '-' + str(mo).zfill(2) + '-*.nc')
        ds = xr.open_mfdataset(fnms)
    else:
        fnms = glob(fnmhead + str(yr) + '-*.nc')
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
    
    fnm = dirData + 'Output/' + str(yr) + '/GFED5NRTcumu' + vnm + '_' + str(yr) + '.csv'
    if not os.path.exists(fnm):
        return None
    else:
        df = pd.read_csv(fnm, index_col=0, parse_dates=True)
    
    return df

def doGFEDregsum(da,yr,mo, day):
    import pandas as pd
    
    daGFED = read_GFEDmask()
    
    sums = {}
    sums['Globe'] = da.sum().item()
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
    
    if df is None:
        # If df is empty, initialize it with df_day
        df_updated = df_day.copy()
    else:
        df_updated = pd.concat([df,df_day],axis=0)
    
    # save updated GFED regional sums to CSV
    fnm = dirData + 'Output/' + str(yr) + '/GFED5NRTcumu' + vnm + '_' + str(yr) + '.csv'
    df_updated.to_csv(fnm)
         
    return df_updated

def readGFED5clim():
    fnmGFED5 = os.path.join(dirData, 'Input', 'table_EM_2002-2022.csv')
    df_all = pd.read_csv(fnmGFED5, index_col=[0, 1], parse_dates=["time"])
    return df_all

def getGFED5climTotal():
    df_all = readGFED5clim()
    EM_Total_mo = df_all.xs('Globe',level='Reg')['Total']  # g C/mon
    EM_Total_mo.name = 'Globe'
    for regnm in GFEDnms[1:]:
        EM_Total_mo_reg = df_all.xs(regnm,level='Reg')['Total']
        EM_Total_mo_reg.name = regnm
        EM_Total_mo = pd.concat([EM_Total_mo,EM_Total_mo_reg],axis=1)
    return EM_Total_mo

def monthly_to_yearly_dayofyear_mean(monthly_series):
    """
    Converts a pandas Series with monthly datetime indices to a DataFrame
    with 'dayofyear' indices and columns representing the daily means
    calculated from the monthly data.

    Args:
        monthly_series (pd.Series): A pandas Series with monthly datetime indices.

    Returns:
        pd.DataFrame: A DataFrame with 'dayofyear' indices (1 to 365 or 366)
                      and columns representing the daily means for each year
                      present in the original monthly series.
    """
    yearly_data = {}
    years = monthly_series.index.year.unique()

    for year in years:
        monthly_data_year = monthly_series[monthly_series.index.year == year]
        if not monthly_data_year.empty:
            # Create a daily index for the entire year
            start_date = pd.to_datetime(f'{year}-01-01')
            end_date = pd.to_datetime(f'{year}-12-31')
            daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
            daily_means = pd.Series(index=daily_index)

            # Calculate and map daily means for each month
            for month_date, monthly_value in monthly_data_year.items():
                first_day_of_month = pd.to_datetime(f'{month_date.year}-{month_date.month}-01')
                last_day_of_month = (first_day_of_month + pd.offsets.MonthEnd(0))
                num_days = (last_day_of_month - first_day_of_month).days + 1
                daily_mean_value = monthly_value / num_days

                # Assign the daily mean to all days of the month
                daily_means[first_day_of_month:last_day_of_month] = daily_mean_value

            # Extract day of year as the new index
            yearly_data[year] = pd.Series(daily_means.values, index=daily_means.index.dayofyear)

    # Concatenate the yearly daily series into a DataFrame
    df_dayofyear_mean = pd.DataFrame(yearly_data)
    return df_dayofyear_mean

def make_daily_emissions_table_yr(yr):
    varnm = 'EM'
    sat = 'CMB'
    ds_fires = xr.open_mfdataset(dirData+'Output/'+str(yr)+'/GFED5NRTeco_'+sat+'_????-??-??.nc')
    da = ds_fires[varnm].sum(dim='lct') 

    # calculate mask 
    daGFED = read_GFEDmask()

    # calculate total CO emissions (in grams/day) in a region 
    print('Processing global sum...')
    ts = da.sum(dim=['lat','lon']).compute().to_pandas().to_frame(name="Globe")
    GFEDnms = ['OCEA', 'BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS', 'AUST']
    for GFEDreg in range(1,15):
        print(f'Processing region {GFEDreg}...')
        masked_data = da.where(daGFED == GFEDreg, drop=False)
        regnm = GFEDnms[GFEDreg]
        ts[regnm] = masked_data.sum(dim=['lat','lon']).compute().to_pandas()

    ts.index = ts.index.dayofyear
    ts.index.name = 'Day_of_Year'
    
    # save to csv
    outfile = dirData + 'Input/DailyRegionalSum/daily_emissions_table_'+str(yr)+'.csv'
    ts.to_csv(outfile)
    

def pltEMfig(df_updated,cumu=False):
    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.lines as mlines 
    import matplotlib.patches as mpatches

    matplotlib.use('Agg')  # Non-interactive backend for PNG/PDF output, which is needed for some servers

    lastdate = df_updated.index[-1] 
    # currentyear = lastdate.year
    # currentyear = int(lastdate[0:4])
    if isinstance(lastdate, datetime.date):
        currentyear = lastdate.year
    elif isinstance(lastdate, str):
        currentyear = int(lastdate[0:4])

    # calculate global sum from climatology data
    dfmean = pd.read_csv(os.path.join(dirData, 'Input/DailyRegionalSum/daily_emissions_table_means_2002-2022.csv'),index_col=0)
    dfallyrs = {}
    for yr in range(2002, currentyear):
        dfallyrs[yr] = pd.read_csv(os.path.join(dirData, 'Input/DailyRegionalSum/daily_emissions_table_'+str(yr)+'.csv'),index_col=0)              
    dfstd = pd.read_csv(os.path.join(dirData, 'Input/DailyRegionalSum/daily_emissions_table_stds_2002-2022.csv'),index_col=0)

    # Set plotting style
    c_other, c_current = '0.6', 'red'
    fig, axes = plt.subplots(ncols=5,nrows=3, figsize=(14, 8))
    axs = axes.ravel()

    # Plot global and regional panels
    for i, reg in enumerate(['Globe'] + GFEDnms[1:]):
        df_updated_reg = df_updated[reg].copy()/1e12
        df_updated_reg.index = pd.to_datetime(df_updated_reg.index, format='mixed').dayofyear
        df_clim_reg  = dfmean[reg]/1e12
        if cumu:
            df_updated_reg = df_updated_reg.cumsum()
            for yr in range(2002, currentyear):
                df_yr = dfallyrs[yr][reg]/1e12
                df_yr = df_yr.cumsum()
                color = c_other
                if yr == currentyear - 1:
                    color = '#f56f42'
                if yr == currentyear - 2:
                    color = '#faa302'
                axs[i].plot(df_yr.index, df_yr, color=color)
            tt = f'Cumulative global and regional fire emissions (Tg C) \n updated on {lastdate}'
            axs[i].axvline(x=pd.to_datetime(lastdate).day_of_year+1, ls='--', color='0.5', lw=0.5)
        else:
            df_clim_reg  = dfmean[reg]/1e12
            df_std_reg  = dfstd[reg]/1e12
            axs[i].fill_between(dfmean.index, df_clim_reg - df_std_reg, df_clim_reg + df_std_reg, color=c_other, alpha=0.3)
            axs[i].plot(dfmean.index, df_clim_reg, color='black', linewidth=1)
            tt = f'Daily global and regional fire emissions (Tg C / day) \n updated on {lastdate}'
        df_updated_reg.plot(ax=axs[i],lw=2,legend=False,color=c_current)
        axs[i].text(0.02,0.98, reg, transform=axs[i].transAxes, ha="left", va="top", 
        fontsize=12, color="k")
        axs[i].set_ylim(0, None)

    if cumu:
        red_patch = mlines.Line2D([],[],color='red', label=str(currentyear))
        orange_patch = mlines.Line2D([],[],color='#f56f42', label=str(currentyear-1))
        yellow_patch = mlines.Line2D([],[],color='#faa302', label=str(currentyear-2))
        gray_patch = mlines.Line2D([],[],color='0.6', label='2002-2022')
        _=fig.legend(loc='upper left', ncol=2, fontsize=10, bbox_to_anchor=(0.01,0.99), frameon=False,
                        handles=[red_patch,orange_patch,yellow_patch,gray_patch])
    else:
        red_patch = mlines.Line2D([],[],color='red', label=str(currentyear))
        black_patch = mlines.Line2D([],[],color='k', label='2002-2022 mean')
        gray_patch = mpatches.Patch(color=c_other, alpha=0.3, label='2002-2022 range')
        _=fig.legend(loc='upper left', ncol=2, fontsize=10, bbox_to_anchor=(0.01,0.99), frameon=False,
                        handles=[red_patch,black_patch,gray_patch])        
        
    fig.text(x=0.5,y=0.01,s='Day of the Year', ha='center', va='bottom',fontsize=10)
    fig.suptitle(tt)
    _=plt.tight_layout(rect=[0, 0.03, 1, 1])
    print(f'Figure updated for {lastdate}')
    
    return fig

def updatetsdata(yr, mo, day, vnm='EM', sat=None):
    """
    Update the cumulative emissions data by appending daily emission sums
    from the day after the last saved date up to the specified target date.

    Parameters:
    -----------
    yr : int
        Target year to update up to.
    mo : int
        Target month.
    day : int
        Target day.
    vnm : str, optional
        Variable name to read and update (default is 'EM').

    Returns:
    --------
    df : pandas.DataFrame
        Updated cumulative emissions DataFrame.
    """
    
    # Load existing cumulative emissions data
    df = readcumudata(yr, vnm=vnm)

    # Safely parse the last date in the index to a datetime.date object
    if df is None:
        last_date = datetime.date(yr, 1, 1) - datetime.timedelta(days=1)  # set to day before Jan 1 of target year
    else:
        last_date_str = df.index.max()
        last_date = pd.to_datetime(last_date_str).date()

    # Define the target end date
    target_date = datetime.date(yr, mo, day)

    # If data is already current, return unchanged
    if last_date >= target_date:
        print("Data is already up to date.")
        return df

    # Loop from the day after last_date up to and including target_date
    current_date = last_date + datetime.timedelta(days=1)
    while current_date <= target_date:
        print(f"Processing data for {current_date.isoformat()}...")

        # Read daily emissions
        da = readGFED5eco(current_date.year, mo=current_date.month, day=current_date.day, vnm=vnm, sat=sat)

        # Aggregate daily emissions data
        if da is not None:
            df_day = doGFEDregsum(da, current_date.year, current_date.month, current_date.day)

            # Update cumulative DataFrame
            df = updateGFEDregsum(df, df_day, current_date.year, vnm)

        # Move to next day
        current_date += datetime.timedelta(days=1)

    return df

def generatetsfig(yr, vnm, df_updated=None):
    # If df_updated is not provided, read the updated cumulative data
    if df_updated is None:
        df_updated = readcumudata(yr, vnm='EM')

    # # read climatological cumulative data
    # EM_Total_mo = getGFED5climTotal()

    # generate updated figs
    figdaily = pltEMfig(df_updated, cumu=False)
    figcumu = pltEMfig(df_updated, cumu=True)

    # save the figure
    fig_path = dirData + 'Output/' + str(yr) + '/' 
    fnmdaily = fig_path+'GFED5NRTdaily' + vnm + '_' + str(yr) + '.png'
    fnmcumu = fig_path+'GFED5NRTcumu' + vnm + '_' + str(yr) + '.png'
    figdaily.savefig(fnmdaily)
    figcumu.savefig(fnmcumu)

    return fnmdaily, fnmcumu

def generatetsfig_fordate(yr, mo, day, vnm='EM', sat=None):
    """
    Generate daily and cumulative emission time-series figures for an arbitrary date.

    Unlike ``generatetsfig()``, which reads from a presaved cumulative CSV via
    ``readcumudata()``, this function computes the required DataFrames on the
    fly from the individual daily GFED5NRT NetCDF files so that figures can be
    produced for any date that has emission data available.

    Parameters
    ----------
    yr : int
        Year of the target date.
    mo : int
        Month of the target date.
    day : int
        Day of the target date.
    vnm : str, optional
        Variable name to read (default is 'EM').
    sat : str or None, optional
        Satellite identifier passed to ``readGFED5eco()`` (e.g. 'VNP', 'VJ1',
        'CMB').  If None, the default behaviour of ``readGFED5eco()`` is used.

    Returns
    -------
    fnmdaily : str
        File path of the saved daily-emissions figure.
    fnmcumu : str
        File path of the saved cumulative-emissions figure.
    """

    target_date = datetime.date(yr, mo, day)

    # ------------------------------------------------------------------
    # Check for a previously saved df_all; if available, load it directly
    # ------------------------------------------------------------------
    fig_path = dirData + 'Output/' + str(yr) + '/'
    mkdir(fig_path)
    fnm_csv = fig_path + 'GFED5NRTcumu' + vnm + '_' + str(yr) + '_' + strymd(yr, mo, day) + '.csv'

    if os.path.exists(fnm_csv):
        print(f"Reading cached regional-sum data from {fnm_csv}")
        df_all = pd.read_csv(fnm_csv, index_col=0, parse_dates=True)
    else:
        # ------------------------------------------------------------------
        # Step 1 & 2: Read daily emissions from Jan 1 to the target date and
        #             compute regional sums for every day using doGFEDregsum()
        # ------------------------------------------------------------------
        df_all = None
        current_date = datetime.date(yr, 1, 1)

        while current_date <= target_date:
            print(f"Processing {current_date.isoformat()}...")

            # Read the daily gridded emission for this date
            da = readGFED5eco(
                current_date.year,
                mo=current_date.month,
                day=current_date.day,
                vnm=vnm,
                sat=sat,
            )

            if da is not None:
                # Step 3: Calculate regional sums for this day
                df_day = doGFEDregsum(da, current_date.year, current_date.month, current_date.day)

                # Accumulate into a single DataFrame (rows = dates, cols = regions)
                if df_all is None:
                    df_all = df_day.copy()
                else:
                    df_all = pd.concat([df_all, df_day], axis=0)

            current_date += datetime.timedelta(days=1)

        if df_all is None:
            print(f"No emission data found for {yr} up to {target_date.isoformat()}.")
            return None, None

        # Save df_all to CSV for future reuse
        df_all.to_csv(fnm_csv)
        print(f"Regional-sum data saved to {fnm_csv}")

    # ------------------------------------------------------------------
    # Step 4: Generate the two figures using pltEMfig()
    # ------------------------------------------------------------------
    figdaily = pltEMfig(df_all, cumu=False)
    figcumu  = pltEMfig(df_all, cumu=True)

    # ------------------------------------------------------------------
    # Step 5: Save the figures
    # ------------------------------------------------------------------
    fnmdaily = fig_path + 'GFED5NRTdaily' + vnm + '_' + str(yr) + '_' + strymd(yr, mo, day) + '.png'
    fnmcumu  = fig_path + 'GFED5NRTcumu'  + vnm + '_' + str(yr) + '_' + strymd(yr, mo, day) + '.png'
    figdaily.savefig(fnmdaily)
    figcumu.savefig(fnmcumu)

    print(f"Figures saved:\n  daily: {fnmdaily}\n  cumu:  {fnmcumu}")
    return fnmdaily, fnmcumu

def uploadtsfigs(fnmdaily, fnmcumu, UCI=True, WUR=True):
    # upload figures to UCI sftp server
    if UCI:
        sftp_upload(userconfig.ftpurl_UCI, userconfig.ftpun_UCI, userconfig.ftppw_UCI,
            fnmdaily, remote_dir=userconfig.ftpdir_UCI, ftpport=userconfig.ftpport_UCI)    
        sftp_upload(userconfig.ftpurl_UCI, userconfig.ftpun_UCI, userconfig.ftppw_UCI,
            fnmcumu, remote_dir=userconfig.ftpdir_UCI, ftpport=userconfig.ftpport_UCI)
    
    # upload figures to WUR sftp server
    if WUR:
        sftp_upload(userconfig.ftpurl_WUR, userconfig.ftpun_WUR, userconfig.ftppw_WUR,
            fnmdaily, remote_dir=userconfig.ftpdir_WUR, ftpport=userconfig.ftpport_WUR)    
        sftp_upload(userconfig.ftpurl_WUR, userconfig.ftpun_WUR, userconfig.ftppw_WUR,
            fnmcumu, remote_dir=userconfig.ftpdir_WUR, ftpport=userconfig.ftpport_WUR)
    
def update_to_now_figs(vnm='EM', genfig=True, uplfig=True, date_now=None):
    # The figure from CMB is produced
    
    # Get the latest GFED5NRT date
    if date_now is None:
        date_now = datetime.datetime.now().date()
    date_GFED5 = get_GFED5_lastday(date_now.year)
    
    # update the cumulative data
    _ = updatetsdata(date_GFED5.year, date_GFED5.month, date_GFED5.day, vnm=vnm, sat='CMB')

    # generate the updated cumulative figure
    if genfig:
        fnmdaily, fnmcumu = generatetsfig(date_now.year, vnm)
        if uplfig:
            uploadtsfigs(fnmdaily, fnmcumu, UCI=True, WUR=True)
    
# Optional step : Convert GFED5NRT from daily to monthly (may be deleted later)
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
# GFEDextNRT bulk run functions
# ----------------------------------------------------------------------------------------------------

def run_1day(yr, mo, day, IMG=True, upd=False, sat='VNP'):
    """Generates GFED5NRT data for a single day.

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

    print("...downloading data from NASA server and creating fire location data...")
    if IMG:
        dlflag = make_VNP14IMGDL(yr, mo, day, upd=upd, sat=sat)
    else:
        print("...using pre-downloaded VNP14IMGML data...")
        dlflag = True

    if not dlflag:
        print(f"Warning: No data available for {yr}-{mo}-{day}.")
        return
    
    print("...reading and preprocessing VIIRS active fire location data...")
    dfFC = readpreprocess_DL(yr, mo, day, IMG=IMG, sat=sat)

    print("...recording VIIRS active fires in each 500m grid...")
    processor = MODISTileProcessor()
    nVAFC = recordVAF500m(dfFC, yr, mo, day=day, sat=sat, processor=processor)

    print("...recording VIIRS active fires in each 0.25deg grid...")
    if nVAFC > 0:
        ds_VAF_types,ds_VAF_bioms = cal_VAFp25_alltiles_1day(yr, mo, day, sat=sat)
        sav_VAFp25_alltiles_1day(ds_VAF_types,ds_VAF_bioms,yr,mo,day,sat=sat)

        print("...calculating scaled BA for each 0.25deg grid...")
        cal_BA_scled_day(yr, mo, day, sat=sat)
        print("...calculating scaled EM for each 0.25deg grid...")
        cal_EM_scled_day(yr, mo, day, sat=sat)

        print("...generating GFED5NRTeco (16-class combined VAF, BA, EM) data...")
        fnmeco = make_GFED5eco(yr, mo, day,sat=sat)
        upload_file_WUR(fnmeco)

        print("...generating GFED5NRTspe (all-species EM data)...")
        fnmspe = make_EMspecies(yr, mo, day,sat=sat)
        upload_file_WUR(fnmspe)

    else:
        print(f"No active fires were detected by VIIRS for {yr}-{mo}-{day}")

def get_GFED5_lastday(yr):
    """Determines the latest date for which GFED5NRT data has been created.

    Args:
        yr (int): The year for which to check the latest date.

    Returns:
        datetime.date or None: The latest date as a datetime.date object if files are found,
                               otherwise None.
    """
    import os
    from glob import glob
        
    # Construct the directory path for the specified data type
    dirGFED = os.path.join(dirData, 'Output', str(yr))

    # List all `GFED5NRTeco` files in the directory
    all_paths = glob(dirGFED+'/GFED5NRTeco*.nc',)
    files = [os.path.basename(path) for path in all_paths if os.path.isfile(path)]

    if not files:
        print(f"Starting from the start of {yr}")
        return datetime.date(yr-1, 12, 31) # No matching files found

    # Extract dates from file names.
    dates = []
    for file in files:
        try:
            dates.append(datetime.datetime.strptime(file[-13:-3], '%Y-%m-%d').date())
        except ValueError:
            # Skip files that don't match the expected naming convention
            continue

    if not dates:
        print(f"Starting from the start of {yr}")
        return datetime.date(yr-1, 12, 31) # No valid dates extracted from filenames

    # Find and return the latest date
    latest_date = max(dates)
    print(f"Last GFED5 data available up to: {latest_date}")
    return latest_date

def clean_files_year(year):
    delete_subdirs_and_files(dirData+'Input/VNP14IMG/'+str(year))
    delete_subdirs_and_files(dirData+'Input/VNP14IMG_NRT/'+str(year))
    delete_subdirs_and_files(dirData+'Input/VJ114IMG/'+str(year))
    delete_subdirs_and_files(dirData+'Input/VJ114IMG_NRT/'+str(year))
    delete_subdirs_and_files(dirData+'Intermediate/VNP500m/'+str(year))
    delete_subdirs_and_files(dirData+'Intermediate/VJ1500m/'+str(year))

def combineVNPVJ1_1day(yr,mo,day):
    """ combine GFED5e eco and spe from VNP and VJ1 """
    import shutil

    print(f"Combining GFED5NRT data for {yr}-{mo}-{day} from VNP and VJ1...")
    fnm_eco_VNP, fnm_eco_VJ1, fnm_eco_CMB, fnm_spe_VNP, fnm_spe_VJ1, fnm_spe_CMB = get_GFED5e_allfile_paths(yr, mo, day)

    if os.path.exists(fnm_eco_VNP) and (not os.path.exists(fnm_eco_VJ1)):
        shutil.copyfile(fnm_eco_VNP, fnm_eco_CMB)
        shutil.copyfile(fnm_spe_VNP, fnm_spe_CMB)       
    elif os.path.exists(fnm_eco_VJ1) and (not os.path.exists(fnm_eco_VNP)): 
        shutil.copyfile(fnm_eco_VJ1, fnm_eco_CMB)
        shutil.copyfile(fnm_spe_VJ1, fnm_spe_CMB)     
    else:
        # use simple average to combine all dataarrays in VNP and VJ1 files
        combine_files_by_average(fnm_eco_VNP, fnm_eco_VJ1, fnm_eco_CMB)
        combine_files_by_average(fnm_spe_VNP, fnm_spe_VJ1, fnm_spe_CMB)

    # push combined files to WUR
    upload_file_WUR(fnm_eco_CMB)
    upload_file_WUR(fnm_spe_CMB)

def update_to_now():
    """Wrapper function to update GFED5NRT data to the latest available VIIRS data.

    This function checks the last date for which GFED5NRT data exists. If the
    current date is newer than the last available GFED5NRT data, it iteratively
    calls `run_1day` for each missing day to update the dataset.
    """

    # Get the current date and the last available date of GFED5NRT EM data
    date_now = datetime.datetime.now().date()
    year_now = date_now.year
    date_GFED5 = get_GFED5_lastday(year_now)
        
    # If the last availabe date is before the current date, call `run_1day()` for each day between the two dates
    if date_now > date_GFED5:
        print(f"VIIRS data is newer than GFED5 data, updating to {date_now}")
        date_difference = date_now - date_GFED5

        # Iterate through each day that needs to be updated
        for day_offset in range(1, date_difference.days + 1):
            date_run = date_GFED5 + datetime.timedelta(days=day_offset)
            print(f"Processing update for {date_run.year}-{date_run.month}-{date_run.day}...")
            # Always use IMG option for operational update
            run_1day(date_run.year, date_run.month, date_run.day, IMG=True, upd=True, sat='VNP')
            run_1day(date_run.year, date_run.month, date_run.day, IMG=True, upd=True, sat='VJ1')
            combineVNPVJ1_1day(date_run.year, date_run.month, date_run.day)
    else:
        print("GFED5 data is already up to date with VIIRS data.")
        
    # Clean files
    print("clean daily data")
    clean_files_year(year_now)
    
def run_1mon(yr, mo, IMG=False, upd=False):
    ''' generate GFED5NRT for all days in a single month: call `run_1day()` for each day
    '''
    import calendar
    from tqdm import trange

    ndays = calendar.monthrange(yr, mo)[1]
    for day in trange(1, ndays+1):
        run_1day(yr, mo, day, IMG=IMG, upd=upd)

# ----------------------------------------------------------------------------------------------------
# GFEDextNRT remedy run functions
# ----------------------------------------------------------------------------------------------------

def run_1day_remedy(yr, mo, day, IMG=True, upd=False, sat='VNP'):
    """Generates GFED5NRT data for a single day retrospectively. No data uploading is needed.

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

    print("...downloading data from NASA server and creating fire location data...")
    if IMG:
        dlflag = make_VNP14IMGDL(yr, mo, day, upd=upd, sat=sat)
    else:
        print("...using pre-downloaded VNP14IMGML data...")
        dlflag = True

    if not dlflag:
        print(f"Warning: No data available for {yr}-{mo}-{day}.")
        return
    
    print("...reading and preprocessing VIIRS active fire location data...")
    dfFC = readpreprocess_DL(yr, mo, day, IMG=IMG, sat=sat)

    print("...recording VIIRS active fires in each 500m grid...")
    processor = MODISTileProcessor()
    nVAFC = recordVAF500m(dfFC, yr, mo, day=day, sat=sat, processor=processor)

    print("...recording VIIRS active fires in each 0.25deg grid...")
    if nVAFC > 0:
        ds_VAF_types,ds_VAF_bioms = cal_VAFp25_alltiles_1day(yr, mo, day, sat=sat)
        sav_VAFp25_alltiles_1day(ds_VAF_types,ds_VAF_bioms,yr,mo,day,sat=sat)

        print("...calculating scaled BA for each 0.25deg grid...")
        cal_BA_scled_day(yr, mo, day, sat=sat)
        print("...calculating scaled EM for each 0.25deg grid...")
        cal_EM_scled_day(yr, mo, day, sat=sat)

        print("...generating GFED5NRTeco (16-class combined VAF, BA, EM) data...")
        fnmeco = make_GFED5eco(yr, mo, day,sat=sat)

        print("...generating GFED5NRTspe (all-species EM data)...")
        fnmspe = make_EMspecies(yr, mo, day,sat=sat)

    else:
        print(f"No active fires were detected by VIIRS for {yr}-{mo}-{day}")

def run_yearly_combine(target_yr, target_mo, target_day):
    """
    Loops from Jan 1st of the target year up to the specified date
    and runs the combineVNPVJ1_1day function.
    """
    # 1. Define the start and end dates
    start_date = f"{target_yr}-01-01"
    end_date = f"{target_yr}-{target_mo:02d}-{target_day:02d}"
    
    # 2. Generate the range of dates (inclusive)
    # freq='D' ensures we get every single day
    date_series = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"Starting processing from {start_date} to {end_date}...")
    print(f"Total days to process: {len(date_series)}")

    # 3. Loop and call your function
    for current_date in date_series:
        yr = current_date.year
        mo = current_date.month
        day = current_date.day
        
        try:
            # Formatting strings to show progress (e.g., "Processing 2025-01-01")
            print(f"Processing: {yr}-{mo:02d}-{day:02d}", end='\r')
            
            # Call your specific function
            combineVNPVJ1_1day(yr, mo, day)
            
        except Exception as e:
            print(f"\n[!] Error on {yr}-{mo:02d}-{day:02d}: {e}")
            # Optional: 'continue' to skip errors, or 'break' to stop the whole loop
            continue

    print("\nProcessing complete.")
# ----------------------------------------------------------------------------------------------------
# GFEDext run using standard VIIRS data (ST)
# ----------------------------------------------------------------------------------------------------

# run GFED5ext ST for all days within a month parallelly
def readpreprocess_ML(yr,mo,sat='VNP'):
    ''' read and preprocess the VIIRS monthly active fire location data
    '''
    # read monthly VIIRS fire locations from standard ML data
    df_VNP_month = read_VNP14IMGML(yr,mo,sat=sat)

    # add major LCT
    df_VNP_month = adddfmjLCT(df_VNP_month)

    # add DN flag
    df_VNP_month = addDN(df_VNP_month, NRT=False)

    # add sample adjustment weight
    df_VNP_month = add_wgt_2_VNP(df_VNP_month, mo)

    return df_VNP_month

def day_prun_ST(df_VNP_month, sat, date_tuple):
    """ run GFED5ext ST for 1 day
    """
    # print(f"Type of df_VNP_month: {type(df_VNP_month)}")
    # print(f"sat: {sat}")

    # get yr, mo, day from input date tuple
    yr, mo, day = date_tuple

    # # read daily VIIRS fire locations from standard ML data
    # print("...reading VIIRS active fire locations...")
    # df_VNP_sample = read_VNP14IMGML(yr,mo,day=day, sat=sat)

    # get MODIS tile information using a class object (save running time)
    processor = MODISTileProcessor()   

    # extract daily VIIRS fire locations from monthly data
    df_VNP_day = df_VNP_month[df_VNP_month['YYYYMMDD'].astype(str).str.endswith(str(day).zfill(2))]

    # record fire locations to 500m rasters
    print("...recording VIIRS active fire locations in each 500m grid...")
    nVAFC = recordVAF500m(df_VNP_day, yr,mo,day=day, vhs=[0,35],vvs=[0,17], sat=sat, processor=processor)

    # skip the day if no active fires were detected
    if nVAFC > 0:
        # record fire counts to 0.25 deg cells
        print("...recording VIIRS active fires in each 0.25deg grid...")
        ds_VAF_types,ds_VAF_bioms = cal_VAFp25_alltiles_1day(yr, mo, day, sat=sat)
        sav_VAFp25_alltiles_1day(ds_VAF_types,ds_VAF_bioms,yr,mo,day,sat=sat)

        # calculate scaled BA from AF
        print("...calculating scaled BA for each 0.25deg grid...")
        cal_BA_scled_day(yr, mo, day, sat=sat)

        # calculate scaled EM from scaled BA
        print("...calculating scaled EM for each 0.25deg grid...")
        cal_EM_scled_day(yr, mo, day, sat=sat)

        # generate GFED5ext_eco and GFED5ext_spe data files without uploading
        print("...generating GFED5NRTeco (16-class combined VAF, BA, EM) data...")
        _ = make_GFED5eco(yr, mo, day,sat=sat)
        print("...generating GFED5NRTspe (all-species EM data)...")
        _ = make_EMspecies(yr, mo, day,sat=sat)
    
    else:
        print(f"No active fires were detected by VIIRS for {yr}-{mo}-{day}")

def mo_prun_ST(yr, mo, sat = 'VNP', max_workers = 1):
    """ wrapper to run GFED5ext for all days within a month parallelly 
    """
    from functools import partial
    import concurrent.futures

    print(f"--- Starting Parallel Processing ({datetime.datetime.now().strftime('%H:%M:%S')}) ---")

    # list of days within a month
    start_date = datetime.date(yr, mo, 1)
    _, num_days = calendar.monthrange(yr, mo)
    end_date = datetime.date(yr, mo, num_days)
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append((current_date.year, current_date.month, current_date.day))
        current_date += datetime.timedelta(days=1)    
    print(f"Total dates to process: {len(date_list)}")
    print(date_list)

    # read monthly VIIRS fire locations from standard ML data
    print("...reading and preprocessing VIIRS active fire locations...")
    df_VNP_month = readpreprocess_ML(yr,mo,sat=sat)

    # use functools.partial to fix df_VNP_month and sat arguments
    fixed_day_prun_ST = partial(day_prun_ST, df_VNP_month, sat)

    # Execute daily tasks (day_prun_ST) in parallel using a Process Pool
    start_time = time.time()
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f"Using a maximum of {executor._max_workers} worker processes.")
        results_iterator = executor.map(fixed_day_prun_ST, date_list)
        for result in results_iterator:
            print(result)
            all_results.append(result)
    end_time = time.time()
    
    print("\n--- Summary ---")
    print(f"Parallel execution finished in {end_time - start_time:.2f} seconds.")

# fill in missing days via linear interpolation
def read_missingdays():
    """ read missing VIIRS observation days from csv file
    source: https://firms.modaps.eosdis.nasa.gov/api/missing_data/
    """
    df_missingVIIRS = pd.read_csv(dirData +'Input/standard_missing_data.csv', index_col=0, parse_dates=True)
    df_missing_VNP = df_missingVIIRS[df_missingVIIRS['Satellite (Sensor)'] == 'Suomi NPP (VIIRS)'].index
    df_missing_VJ1 = df_missingVIIRS[df_missingVIIRS['Satellite (Sensor)'] == 'NOAA-20 (VIIRS)'].index

    return df_missing_VNP, df_missing_VJ1

def find_nearest_available_days(yr, mo, day, df_missing):
    """
    Finds the nearest available date with data before and after the missing day.
    
    Args:
        yr, mo, day (int): The date components of the missing day.
        df_missing (pd.DatetimeIndex/list): Collection of ALL missing dates.
        
    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: The previous and next available dates.
    """
    date_missing = pd.Timestamp(year=yr, month=mo, day=day)
    
    # 1. Create a continuous date range for the surrounding period
    # Use 30 days before and after as a safety net, though usually finding nearest is faster
    start_date = date_missing - pd.Timedelta(days=30)
    end_date = date_missing + pd.Timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 2. Identify available dates (dates in the range that are NOT in df_missing)
    # We convert df_missing to a DatetimeIndex for efficient set operations
    if not isinstance(df_missing, pd.DatetimeIndex):
        df_missing = pd.to_datetime(df_missing)

    available_dates = date_range[~date_range.isin(df_missing)]
    
    # 3. Find the nearest preceding and succeeding dates
    
    # Nearest previous available date
    # Look for dates in the available list that are strictly before the missing date
    prev_dates = available_dates[available_dates < date_missing]
    date_prev_found = prev_dates[-1] if not prev_dates.empty else None # Get the latest one (last element)
    
    # Nearest next available date
    # Look for dates in the available list that are strictly after the missing date
    next_dates = available_dates[available_dates > date_missing]
    date_next_found = next_dates[0] if not next_dates.empty else None # Get the earliest one (first element)

    # 4. Apply modification
    # date_prev should be 1 day BEFORE the found nearest available previous date
    date_prev = date_prev_found - pd.Timedelta(days=1) if date_prev_found is not None else None
    
    # date_next should be 1 day AFTER the found nearest available next date
    date_next = date_next_found + pd.Timedelta(days=1) if date_next_found is not None else None

    return date_prev, date_next

def get_GFED5e_file_path(yr, mo, day, sat, product):
    """ Helper function to construct the expected file path. """
    if day == 0: # return monthly file path
        date_str = str(yr)+'-'+str(mo).zfill(2)
    else:
        date_str = strymd(yr, mo, day)
        # The file naming convention is assumed to be: GFED5NRT[product]_[sat]_[ymd].nc

    return dirData + f'Output/{yr}/GFED5NRT{product}_{sat}_{date_str}.nc'

def get_GFED5e_allfile_paths(yr, mo, day):
    fnm_eco_VNP = get_GFED5e_file_path(yr, mo, day, sat='VNP', product='eco')
    fnm_eco_VJ1 = get_GFED5e_file_path(yr, mo, day, sat='VJ1', product='eco')
    fnm_eco_CMB = get_GFED5e_file_path(yr, mo, day, sat='CMB', product='eco') 
    fnm_spe_VNP = get_GFED5e_file_path(yr, mo, day, sat='VNP', product='spe')
    fnm_spe_VJ1 = get_GFED5e_file_path(yr, mo, day, sat='VJ1', product='spe')
    fnm_spe_CMB = get_GFED5e_file_path(yr, mo, day, sat='CMB', product='spe')    
    return fnm_eco_VNP, fnm_eco_VJ1, fnm_eco_CMB, fnm_spe_VNP, fnm_spe_VJ1, fnm_spe_CMB

def fillin_1day(yr, mo, day, date_prev, date_next, sat):
    """ 
    Generates GFED5e eco and spe files for a missing day using linear 
    interpolation from the specified previous and future dates.
    """
    date_missing = pd.Timestamp(year=yr, month=mo, day=day)
    
    if date_prev is None or date_next is None:
        print(f"  Skipping fill for {date_missing.date()}: Cannot find both bounding available dates.")
        return
        
    # Calculate interpolation weights
    td_prev_to_missing = (date_missing - date_prev).days
    td_missing_to_fut = (date_next - date_missing).days
    td_total = (date_next - date_prev).days
    
    # If td_total is 0, something is wrong, but standard formula should handle > 0
    if td_total == 0:
        print(f"  [ERROR] Previous and next date are the same for {date_missing.date()}. Skipping.")
        return

    # W_prev = (Time distance from missing to future) / (Total time distance)
    weight_prev = td_missing_to_fut / td_total
    # W_fut = (Time distance from previous to missing) / (Total time distance)
    weight_fut = td_prev_to_missing / td_total
    
    print(f"  Bounding days: Prev={date_prev.date()}, Next={date_next.date()}. Weights: Prev={weight_prev:.3f}, Next={weight_fut:.3f}.")

    
    # Loop through 'eco' and 'spe' products and perform interpolation
    for product in ['eco', 'spe']:
        fnm_prev = get_GFED5e_file_path(date_prev.year, date_prev.month, date_prev.day, sat, product)
        fnm_fut = get_GFED5e_file_path(date_next.year, date_next.month, date_next.day, sat, product)
        fnm_out = get_GFED5e_file_path(yr, mo, day, sat, product)
        
        try:
            # Check for file existence before loading (robustness)
            if not os.path.exists(fnm_prev) or not os.path.exists(fnm_fut):
                print(f"  [WARNING] Data file missing for {product} on one of the bounding dates. Skipping {product}.")
                continue

            # Open the datasets
            ds_prev = xr.open_dataset(fnm_prev, decode_times=False)
            ds_fut = xr.open_dataset(fnm_fut, decode_times=False)
            
            # Linear Interpolation: ds_interp = (ds_prev * W_prev) + (ds_fut * W_fut)
            ds_interp = (ds_prev * weight_prev) + (ds_fut * weight_fut)
        
            # Update time coordinate
            new_date = datetime.date(yr,mo,day)
            new_time_coord = [np.datetime64(new_date)]
            ds_interp = ds_interp.assign_coords(time=new_time_coord)
            
            # Update attributes and save
            ds_interp.attrs = ds_prev.attrs
            ds_interp.attrs['interpolation_note'] = f'Linear interpolation using {date_prev.date()} and {date_next.date()} data.'
            
            to_netcdf(ds_interp, fnm_out)
            print(f"  -> Saved interpolated {product} file: {os.path.basename(fnm_out)}")
            
            ds_prev.close()
            ds_fut.close()

        except Exception as e:
            print(f"  [ERROR] Failed to process {product} file for {date_missing.date()}: {e}")
            
    return

def fillin_1mon(sat='VNP',m=[2024,6]):
    """ generate GFED5e eco and spe files for days with missing VIIRS observation """

    # read missing VIIRS observation days
    df_missing_VNP, df_missing_VJ1 = read_missingdays()
    if sat == 'VNP':
        df_missing = df_missing_VNP
    elif sat == 'VJ1':
        df_missing = df_missing_VJ1

    fillindays = df_missing[(df_missing.year == m[0]) & (df_missing.month == m[1])].sort_values()
    
    for d in fillindays:
        yr, mo, day = d.year, d.month, d.day
        print(f"\nProcessing missing VIIRS observation for {sat} on {yr}-{mo:02d}-{day:02d}...")
        
        # find nearest available date with data before and after the missing day
        date_prev, date_next = find_nearest_available_days(yr, mo, day, df_missing)
        
        print(yr,mo,day,date_prev,date_next)
        # generate GFED5e eco and spe files for the missing day
        fillin_1day(yr, mo, day, date_prev, date_next, sat=sat)

# combine VNP and VJ1 data to generate final GFED5NRT data
def combine_files_by_average(file_vnp, file_vj1, file_combined):
    """
    Combines two NetCDF files (VNP and VJ1) by calculating a simple average
    of all data variables and saving the result to a new NetCDF file.

    Args:
        file_vnp (str): Path to the NetCDF file from the VNP sensor.
        file_vj1 (str): Path to the NetCDF file from the VJ1 sensor.
        file_combined (str): Path where the averaged NetCDF file should be saved.
    """
    import xarray as xr
    import os

    # 1. Check if input files exist (although the main function checks this, it's good practice here too)
    if not os.path.exists(file_vnp):
        print(f"  [ERROR] VNP file not found: {file_vnp}")
        return
    if not os.path.exists(file_vj1):
        print(f"  [ERROR] VJ1 file not found: {file_vj1}")
        return

    try:
        # 2. Open the datasets
        # We use decode_times=False to avoid issues with non-standard time units in some NetCDF files
        ds_vnp = xr.open_dataset(file_vnp, decode_times=False)
        ds_vj1 = xr.open_dataset(file_vj1, decode_times=False)
        
        # 3. Perform the simple average
        # xarray automatically aligns datasets based on coordinate names and performs arithmetic element-wise.
        # This assumes both datasets are on the same grid (i.e., have the same dimensions and coordinates).
        print(f"  Averaging data from VNP and VJ1...")
        ds_combined = (ds_vnp + ds_vj1) / 2.0
        
        # Optional: Copy global attributes from one of the source files and add a processing note
        ds_combined.attrs = ds_vnp.attrs
        ds_combined.attrs['processing_note'] = 'Combined by simple average of VNP and VJ1 datasets.'

        # 4. Save the combined dataset to a new NetCDF file
        to_netcdf(ds_combined, file_combined)
        print(f"  Successfully saved combined file: {file_combined}")

    except FileNotFoundError as e:
        print(f"  [ERROR] A file was not found: {e}")
    except xr.MergeError as e:
        print(f"  [ERROR] Datasets could not be merged (possibly incompatible dimensions/coordinates): {e}")
    except Exception as e:
        print(f"  [ERROR] An unexpected error occurred during processing: {e}")
    finally:
        # Ensure datasets are closed
        try:
            if 'ds_vnp' in locals():
                ds_vnp.close()
            if 'ds_vj1' in locals():
                ds_vj1.close()
        except NameError:
            pass # Variables were not defined due to an early error
        except Exception as e:
             print(f"  [WARNING] Error closing datasets: {e}")

def combineVNPVJ1(yr, mo):
    """ combine GFED5e eco and spe from VNP and VJ1 """
    import calendar
    import shutil

    # read missing VIIRS observation days
    df_missing_VNP, df_missing_VJ1 = read_missingdays()

    # loop through all days in the month    
    ndays = calendar.monthrange(yr, mo)[1]
    for day in range(1, ndays+1):
        print(f"Combining GFED5NRT data for {yr}-{mo}-{day} from VNP and VJ1...")
        # fnm_eco_VNP = get_GFED5e_file_path(yr, mo, day, sat='VNP', product='eco')
        # fnm_eco_VJ1 = get_GFED5e_file_path(yr, mo, day, sat='VJ1', product='eco')
        # fnm_eco_CMB = get_GFED5e_file_path(yr, mo, day, sat='CMB', product='eco') # Assuming 'CMB' is the satellite code for combined
        # fnm_spe_VNP = get_GFED5e_file_path(yr, mo, day, sat='VNP', product='spe')
        # fnm_spe_VJ1 = get_GFED5e_file_path(yr, mo, day, sat='VJ1', product='spe')
        # fnm_spe_CMB = get_GFED5e_file_path(yr, mo, day, sat='CMB', product='spe') # Assuming 'CMB' is the satellite code for combined
        fnm_eco_VNP, fnm_eco_VJ1, fnm_eco_CMB, fnm_spe_VNP, fnm_spe_VJ1, fnm_spe_CMB = get_GFED5e_allfile_paths(yr, mo, day)
        daystamp = pd.Timestamp(year=yr, month=mo, day=day)
        if (daystamp in df_missing_VNP) and (daystamp not in df_missing_VJ1):
            # copy VJ1 files to combined files
            shutil.copyfile(fnm_eco_VJ1, fnm_eco_CMB)
            shutil.copyfile(fnm_spe_VJ1, fnm_spe_CMB)
        elif (daystamp not in df_missing_VNP) and (daystamp in df_missing_VJ1):
            # copy VNP files to combined files
            shutil.copyfile(fnm_eco_VNP, fnm_eco_CMB)
            shutil.copyfile(fnm_spe_VNP, fnm_spe_CMB)
        else:
            # use simple average to combine all dataarrays in VNP and VJ1 files
            combine_files_by_average(fnm_eco_VNP, fnm_eco_VJ1, fnm_eco_CMB)
            combine_files_by_average(fnm_spe_VNP, fnm_spe_VJ1, fnm_spe_CMB)

def convert2mon(yr,mo,sat='CMB',product='eco'):
    """ combine all daily GFED5e eco and spe files to monthly sums """
    from glob import glob
    date_str = str(yr)+'-'+str(mo).zfill(2)
    fnms = glob(dirData + f'Output/{yr}/GFED5NRT{product}_{sat}_{date_str}-??.nc')
    ds = xr.open_mfdataset(fnms)
    dsmon = ds.sum(dim='time')
    fnmout = dirData + f'Output/{yr}/GFED5NRT{product}_{sat}_{date_str}.nc'
    to_netcdf(dsmon, fnmout)

def convert2mon_all(yr,mo):
    for sat in ['VNP','VJ1','CMB']:
        for product in ['eco','spe']:
            convert2mon(yr,mo,sat=sat,product=product)

def reformat_GFED5NRT_eco(yr,sat='CMB'):
    """
    1. Read all monthly netcdf files generated by the convert2mon() function
    2. Concatenate (along the time dimension) the monthly data within a year to a single file 
    3. Reformat the data to match the specific format
    4. Save to GFED5eco_YYYY_NRT.nc
    """

    months = range(1, 13)
    ds_list = []
    
    # 1. Read all monthly files
    for mo in months:
        date_str = f"{yr}-{mo:02d}"
        fnm = dirData + f'Output/{yr}/GFED5NRTeco_{sat}_{date_str}.nc'
        if os.path.exists(fnm):
            ds_mo = xr.open_dataset(fnm)
            # Add time dimension (first day of the month)
            dt = pd.to_datetime(f"{yr}-{mo:02d}-01")
            ds_mo = ds_mo.expand_dims(time=[dt])
            ds_list.append(ds_mo)
            print(f"Read monthly file {fnm}")
        else:
            print(f"Warning: Missing monthly file {fnm}")

    if not ds_list:
        print(f"No data found for year {yr}")
        return

    # 2. Concatenate along time dimension
    ds = xr.concat(ds_list, dim='time')
    
    # Sort by time just in case
    ds = ds.sortby('time')

    # Keep only BA and EM variables
    ds = ds[['BA', 'EM']]

    # 3. Reformat
    # Dimensions: (lct, time, lat, lon)
    ds = ds.transpose('lct', 'time', 'lat', 'lon')
    
    # Cast variables to float32
    ds['EM'] = ds['EM'].astype(np.float32)
    ds['BA'] = ds['BA'].astype(np.float32)

    # Set variable attributes
    ds.EM.attrs = {
        'long_name': "emissions",
        'standard_name': "EM",
        'unit': "g C/mon",
        # '_FillValue': np.nan
    }
    ds.BA.attrs = {
        'long_name': "burned area",
        'standard_name': "BA",
        'unit': "m2/mon",
        # '_FillValue': np.nan
    }
    
    # Set lct values and attributes
    ds['lct'] = EMLCnms 
    ds.lct.attrs = {
        'long_name': "ecosystem land cover type",
        'standard_name': "lct",
        'source': "GFED5 reclassified types (16-class), Table 2 of van der Werf et al. (2025)"
    }
    
    # Set time values and attributes
    base_date = pd.to_datetime(str(yr)+"-01-01")
    time_days = [(pd.to_datetime(t) - base_date).days for t in ds.time.values]
    ds['time'] = np.array(time_days, dtype='int64')
    ds.time.attrs = {
        'units': "days since "+str(yr)+"-01-01",
        'calendar': "proleptic_gregorian"
    }
    
    # Set lat/lon attributes
    ds.lat.attrs = {
        'units': "degrees_north",
        'axis': "Y",
        'long_name': "latitude",
        'standard_name': "lat",
    }
    ds.lon.attrs = {
        'units': "degrees_east",
        'axis': "X",
        'long_name': "longitude",
        'standard_name': "lon",
        # '_FillValue': np.nan
    }
    
    # Global attributes
    ds.attrs = {
        'long_name': "Global Fire Emissions Database v5 NRT ecosystem",
        'standard_name': "GFED5NRTeco",
        'creation_date': datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        'frequency': "mon",
        'grid': "0.25x0.25 degree latxlon grid",
        'institution': "University of California, Irvine",
        'license': "Data in this file is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License(https://creativecommons.org/licenses/)",
        'nominal_resolution': "0.25x0.25 degree",
        'region': "global",
        'source': "GFED5 NRT",
        'contact': "Yang Chen (yang.chen@uci.edu)"
    }

    # 4. Save to file
    fnmout = dirData + f'Output/{yr}/GFED5NRTeco_{sat}_{yr}.nc'
    to_netcdf(ds, fnmout)
    print(f"Successfully saved {fnmout}")


# ----------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # nowarn()
    # ------------------------------------------------------------------
    # GFED5NRT run
    # -------------
    update_to_now()  # Update GFED5NRT to the latest date with VIIRS data
    update_to_now_figs()
    # make_daily_emissions_table_yr(2025) # make daily emissions table for 2025

    # 1day or 1mon test run
    # run_1day(2025,11,11, IMG=True, upd=True)  # Derive GFED5 betaV BA and EM for 1 day
    # run_1day(2025,1,1, IMG=True, upd=False)
    # run_1mon(2025,1,IMG=True, upd=True)
    # run_1mon(2025,2,IMG=True, upd=True)

    # ------------------------------------------------------------------
    # GFED5e run
    # -----------
    # yr, mo = 2023, 1
    # mo_prun_ST(yr, mo, sat = 'VJ1', max_workers = 4)   # run either VJ1 or VNP

    # fill in missing days
    # fillin_1mon(sat='VNP',m=[2024,5])
    # fillin_1mon(sat='VNP',m=[2024,6])
    # fillin_1mon(sat='VNP',m=[2024,7])
    # fillin_1mon(sat='VNP',m=[2024,9])
    # fillin_1mon(sat='VNP',m=[2024,11])
    # fillin_1mon(sat='VJ1',m=[2023,7])
    # fillin_1mon(sat='VJ1',m=[2023,9])
    # fillin_1mon(sat='VJ1',m=[2024,2])
               
    # yr = 2024  # combine VJ1 and VNP to generate CMB
    # for mo in range(1,12+1):
    #     combineVNPVJ1(yr,mo)

    # yr, mo = 2025, 1  # convert to monthly data
    # for mo in range(1,12+1):
    #     convert2mon_all(yr,mo)

    # yr = 2023  # reformat to annual ecosystem files
    # reformat_GFED5NRT_eco(yr,sat='CMB')

    # ------------------------------------------------------------------
    # Remedy runs
    # ------------------------------------------------------------------
    # yr, mo, day = 2025, 2, 1
    # import calendar
    # for mo in range(9, 11+1):
    #     ndays = calendar.monthrange(yr, mo)[1]
    #     for day in range(1, ndays+1):
    #         run_1day_remedy(yr, mo, day, IMG=True, upd=False, sat='VJ1')  # ~12 min/day -> 6 hour/mon -> 3 day/yr
    # yr, mo = 2025, 12
    # for day in range(1, 15+1):
    #     run_1day_remedy(yr, mo, day, IMG=True, upd=False, sat='VJ1')
    
