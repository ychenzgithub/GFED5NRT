dirProj = '/home/ychen17/GFED5eNRT/'

import xarray as xr
import pandas as pd
from typing import Optional

def tif2arr(fnm):
    ''' Read numpy array from a tif file (first layer). This is only used in `read_GFEDmask()` for reading a tif file.

    Parameters
    ----------
    fnm : str
        The Geotiff file name

    Returns
    -------
    value : np array
        The 2-D array containing the first-layer data of the tif file
    '''
    import os
    from osgeo import gdal
    if os.path.exists(fnm):
        ds = gdal.Open(fnm)
        value = ds.GetRasterBand(1).ReadAsArray()
        ds = None

        return value
    else:
        return None
    
def getGFEDp25mask():
    import numpy as np
    import xarray as xr

    arrGFED = tif2arr(dirProj + 'Input/GFEDregions_025d.tif')
    daGFED = xr.DataArray(arrGFED[::-1,:],
                    coords={'lat':np.linspace(-89.875,89.875,720),'lon':np.linspace(-179.875,179.875,1440)},
                    dims=['lat','lon'])

    # convert GFED lon from -180~180 to 0~360
    lon_025_positive = daGFED.lon.values.copy()
    lon_025_positive[lon_025_positive < 0] += 360
    daGFED_360 = daGFED.copy()
    daGFED_360 = daGFED_360.assign_coords(lon=lon_025_positive)
    daGFED_360 = daGFED_360.sortby('lon')

    return daGFED, daGFED_360

def make_spe_regsum(varnm='CO', sat='CMB', yr= 2023):
    import xarray as xr

    ds_fires = xr.open_mfdataset(dirProj+'Output/'+str(yr)+'/GFED5eNRTspe_'+sat+'_????-??-??.nc')
    da = ds_fires[varnm]  # g C/day 

    # calculate mask 
    daGFED, _ = getGFEDp25mask()

    # calculate total CO emissions (in grams/day) in a region 
    print('Processing global sum...')
    CO_gram = da.sum(dim=['lat','lon']).compute().to_pandas().to_frame(name="Globe")
    GFEDnms = ['OCEA', 'BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS', 'AUST']
    for GFEDreg in range(1,15):
        print(f'Processing region {GFEDreg}...')
        masked_data = da.where(daGFED == GFEDreg, drop=False)
        regnm = GFEDnms[GFEDreg]
        CO_gram[regnm] = masked_data.sum(dim=['lat','lon']).compute().to_pandas()

    # save to csv
    outfile = dirProj + 'Output/daily_table_GFED51ext'+sat+'_'+varnm+'_'+str(yr)+'.csv'
    CO_gram.to_csv(outfile)

def make_spe_regsum_optimized(varnm='CO', sat='CMB', yr=2023):
    """ Run make_sep_regsum() function without reading all data (in order to reduce memory usage)
    """
    import glob # Needed to find all files for iteration
    import xarray as xr
    import pandas as pd

    # --- Setup ---
    # Assume dirProj and getGFEDp25mask() are defined outside this function
    # or passed as arguments.
    
    # Calculate GFED mask only once (it's independent of time/data)
    # The mask contains region numbers (1 to 14) for each grid cell.
    daGFED, _ = getGFEDp25mask()
    GFEDnms = ['OCEA', 'BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 
               'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS', 'AUST']
    
    # List to hold the resulting daily tables (DataFrames) for each month
    monthly_results = []
    
    # --- Monthly Processing Loop ---
    
    # 1. Find all monthly file patterns (YYYY-MM-??) for the given year
    # This assumes the files are structured by YYYY/GFED5eNRTspe_..._YYYY-MM-DD.nc
    # The glob pattern will find all unique YYYY-MM prefixes
    file_patterns = sorted(glob.glob(dirProj + f'Output/{yr}/GFED5eNRTspe_{sat}_{yr}-??-??.nc'))
    
    # Extract unique YYYY-MM identifiers to iterate over months
    for mo in range(1, 13):
        month_str = str(yr)+'-'+str(mo).zfill(2)
        print(f'Processing month: {month_str}')

        # 2. Open only the files for the current month
        # This significantly reduces the initial memory footprint
        file_path_pattern = dirProj + f'Output/{yr}/GFED5eNRTspe_{sat}_{month_str}-??.nc'
        
        # Use open_mfdataset for the monthly files
        print(file_path_pattern)
        ds_month = xr.open_mfdataset(file_path_pattern)
        da_month = ds_month[varnm]  # g C/day
        
        # DataFrame to store the results for the current month
        CO_gram_month = pd.DataFrame()
        
        # --- Region-wise Calculation ---
        
        # Calculate global sum
        print(f'  - Calculating global sum for {month_str}...')
        # Note: .compute() forces the actual calculation on the chunked data
        global_sum = da_month.sum(dim=['lat', 'lon']).compute().to_pandas().to_frame(name="Globe")
        CO_gram_month = global_sum # Initialize the monthly DataFrame

        # Calculate regional sums (GFED regions 1 to 14)
        for GFEDreg in range(1, 15):
            regnm = GFEDnms[GFEDreg]
            print(f'  - Calculating region {GFEDreg} ({regnm}) for {month_str}...')
            
            # Mask the data for the specific region
            # daGFED has the mask, da_month has the daily data
            masked_data = da_month.where(daGFED == GFEDreg, drop=False)
            
            # Calculate sum for the region across all days in the month
            # The result is a time series (daily sums for the month)
            CO_gram_month[regnm] = masked_data.sum(dim=['lat', 'lon']).compute().to_pandas()
            
        # 3. Append the resulting DataFrame for the month
        monthly_results.append(CO_gram_month)
        
        # Close the dataset for the month to free up memory (good practice)
        ds_month.close()

    # --- Final Step: Combine and Save ---

    if monthly_results:
        print('Combining monthly results...')
        # Concatenate all monthly DataFrames into a single, combined DataFrame
        CO_gram_combined = pd.concat(monthly_results)
        
        # Sort by index (date) to ensure correct order
        CO_gram_combined = CO_gram_combined.sort_index()
        
        # Save the combined DataFrame to CSV
        outfile = dirProj + f'Output/daily_table_GFED51ext{sat}_{varnm}_{yr}.csv'
        print(f'Saving combined table to: {outfile}')
        CO_gram_combined.to_csv(outfile)
        
        return CO_gram_combined
    else:
        print(f'No data found for {yr}.')
        return None

def make_eco_regsum(varnm='EM', sat='CMB', yr= 2023):
    import xarray as xr

    ds_fires = xr.open_mfdataset(dirProj+'Output/'+str(yr)+'/GFED5eNRTeco_'+sat+'_????-??-??.nc')
    da = ds_fires[varnm].sum(dim='lct') 

    # calculate mask 
    daGFED, _ = getGFEDp25mask()

    # calculate total CO emissions (in grams/day) in a region 
    print('Processing global sum...')
    ts = da.sum(dim=['lat','lon']).compute().to_pandas().to_frame(name="Globe")
    GFEDnms = ['OCEA', 'BONA', 'TENA', 'CEAM', 'NHSA', 'SHSA', 'EURO', 'MIDE', 'NHAF', 'SHAF', 'BOAS', 'CEAS', 'SEAS', 'EQAS', 'AUST']
    for GFEDreg in range(1,15):
        print(f'Processing region {GFEDreg}...')
        masked_data = da.where(daGFED == GFEDreg, drop=False)
        regnm = GFEDnms[GFEDreg]
        ts[regnm] = masked_data.sum(dim=['lat','lon']).compute().to_pandas()

    # save to csv
    outfile = dirProj + 'Output/daily_table_GFED51ext'+sat+'_'+varnm+'_'+str(yr)+'.csv'
    ts.to_csv(outfile)

def make_eco_regsum_optimized(
    varnm: str = 'EM', 
    sat: str = 'CMB', 
    yr: int = 2023, 
    chunk_size: Optional[dict] = {'time': 30} # Define chunking strategy
):
    """
    Calculates the daily global and regional sums of a specified fire emission 
    variable efficiently using Dask for large datasets.
    """

    # 1. Load data with Dask/Chunking (Crucial change)
    print(f'Loading data for {yr} using Dask...')
    file_pattern = f'{dirProj}Output/{yr}/GFED5eNRTeco_{sat}_????-??-??.nc'
    
    # Use chunks=chunk_size to enable Dask, which loads data lazily.
    # The computation is only triggered later by .compute().
    ds_fires = xr.open_mfdataset(
        file_pattern, 
        combine='by_coords', 
        chunks=chunk_size # <-- This enables Dask processing
    )
    
    # Sum over the 'lct' (Land Cover Type) dimension
    # da is now a Dask-backed DataArray (a blueprint for computation)
    da = ds_fires[varnm].sum(dim='lct')
    
    # 2. Calculate the regional mask (assumed to fit in memory)
    daGFED, _ = getGFEDp25mask()

    if 'time' in daGFED.dims:
        daGFED = daGFED.isel(time=0, drop=True)
    
    # 3. Define region names
    GFED_REG_MAP = {
        1: 'BONA', 2: 'TENA', 3: 'CEAM', 4: 'NHSA', 5: 'SHSA', 
        6: 'EURO', 7: 'MIDE', 8: 'NHAF', 9: 'SHAF', 10: 'BOAS', 
        11: 'CEAS', 12: 'SEAS', 13: 'EQAS', 14: 'AUST'
    }
    
    # 4. Vectorized Global and Regional Summation (Dask handles the work)
    print('Processing global and regional sums in parallel...')
    
    # A. Global Sum
    # Summing across all spatial dimensions
    da_globe = da.sum(dim=['lat','lon'])
    
    # B. Regional Sum: Groupby is still the fastest method
    # da_regional is a Dask-backed DataArray
    da_regional = da.groupby(daGFED).sum(dim='stacked_lat_lon', skipna=True)
    
    # A. Expand the global data to have a 'group' dimension of size 1 (index 0)
    da_globe_expanded = da_globe.expand_dims('group').assign_coords(group=[0])

    # B. Use xarray.concat to stack along the existing 'group' dimension
    # We reorder da_regional to ensure 'group' is the dimension we stack along.
    da_combined = xr.concat(
        [da_globe_expanded, da_regional.rename(group='group')], # Ensure dim name is consistent
        dim='group' # Stack along the existing 'group' dimension
    )

    # 5. Trigger the Dask computation and convert to pandas
    print('Executing computation (this is where Dask reads and processes chunks)...')
    # This is the single line that triggers all I/O and parallel calculation.
    ts_array = da_combined.compute()

    # 6. Convert to Pandas DataFrame
    # Need to manually create the column names as the concatenation is complex
    column_names = {0: 'Globe'}
    column_names.update(GFED_REG_MAP)

    # Convert the computed xarray to pandas and rename columns
    ts = ts_array.to_pandas().T.rename(columns=column_names)

    # 7. Save to CSV
    outfile = f'{dirProj}Output/daily_table_GFED51ext{sat}_{varnm}_{yr}.csv'
    print(f'Saving results to {outfile}')
    ts.to_csv(outfile)
    
    return ts



def check_gfed_availability(directory, sat = 'VNP', tp='eco'):
    import pandas as pd
    from pathlib import Path
    import xarray as xr
    
    base_path = Path(directory)
    
    # 1. Identify all matching files
    # Pattern: GFED5eNRTeco_VNP_2025-??-??.nc
    files = sorted(list(base_path.glob("GFED5eNRT"+tp+"_"+sat+"_2025-??-??.nc")))
    
    if not files:
        print(f"No files matching the pattern were found in: {directory}")
        return

    # 2. Extract and sort dates
    # Stem is the filename without .nc (e.g., 'GFED5eNRTeco_VNP_2025-08-28')
    # We split by '_' and take the last part
    date_strings = sorted([f.stem.split('_')[-1] for f in files])
    dates = pd.to_datetime(date_strings)

    # 3. Identify Latest Dates
    latest_date = dates.max().strftime('%Y-%m-%d')
    # Get the last 5 unique dates found
    recent_5 = dates.unique()[-5:].strftime('%Y-%m-%d').tolist()

    print(f"--- Data Availability Report: {directory} ---")
    print(f"Total files found: {len(files)}")
    print(f"Earliest data:    {dates.min().strftime('%Y-%m-%d')}")
    print(f"Latest data:      **{latest_date}**")
    print(f"5 Most Recent:    {', '.join(recent_5)}")
    print("-" * 45)

    # 4. Check for Missing Dates in the sequence
    full_range = pd.date_range(start=dates.min(), end=dates.max())
    missing_dates = full_range.difference(dates)

    if not missing_dates.empty:
        print(f"[!] GAPS DETECTED: {len(missing_dates)} missing dates")
        # Group missing dates for easier reading
        for date in missing_dates.strftime('%Y-%m-%d'):
            print(f"    - Missing: {date}")
    else:
        print("[✓] Perfect sequence! No missing dates found between start and end.")

if __name__ == "__main__":
    # make_regsum(varnm='CO', yr= 2023, sat='CMB')
    # make_regsum(varnm='CO', yr= 2023, sat='VNP')
    # make_regsum_optimized(varnm='CO', yr= 2023, sat='VNP')
    # make_regsum(varnm='CO', yr= 2023, sat='VJ1')

    # make_eco_regsum_optimized(varnm='BA', sat='CMB', yr= 2023)
    # make_eco_regsum_optimized(varnm='BA', sat='CMB', yr= 2024)
    
    check_gfed_availability(dirProj+'Output/2025', sat='VNP',tp='eco')
    check_gfed_availability(dirProj+'Output/2025', sat='VNP',tp='spe')
    check_gfed_availability(dirProj+'Output/2025', sat='VJ1',tp='eco')