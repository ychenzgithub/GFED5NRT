# GFED5 Extension NRT Version (GFED5eNRT) Burned Area and Emissions Calculation

This Python script is a standalone tool designed to calculate the GFED5 extension NRT (Near Real-Time) version of burned area (BA) and emissions (EM) at a 0.25-degree resolution. It leverages VIIRS 375m active fire data and employs a 2-step scaling approach.

## Purpose

The primary goal of this script is to:
- Convert daily VIIRS active fire numbers (VAF) to daily burned area (GFED5eNRT BA) using pre-derived effective fire area (EFA) scalars.
- Combine pre-calculated fuel consumption (FC) scalars with GFED5eNRT BA to derive daily emissions (GFED5eNRT EM).

It automates the process of searching for days needing updates, reading and updating VIIRS active fire data, recording VAF, and applying scaling factors.

## Directory Structure

The script expects and generates data within the following directory structure relative to `dirData` (which is set to `~/GFED5eNRT/`):

- `Code/`: Contains the Python script itself.
- `Input/`: Stores all necessary input data files.
- `Intermediate/`: Used for temporary data generated during processing (e.g., 500m VAF data).
- `Output/`: Stores the final GFED5eNRT BA and EM data.

## Key Inputs

The `Input/` directory contains various datasets crucial for the calculations:

- `Earthdata_token.txt`: NASA Earthdata token for data access.
- `GFED51VFA_regtp.csv`: Scalar for VAF to BA conversion.
- `GFED51FC_regtp.csv`: Scalar for BA to EM conversion.
- `VIIRSsplwgt_2019-2021.csv`: Weights for scan angle correction.
- `GFED51_EF.csv`: Table of emission factors.
- `MaskDefonly/`: 500m Deforestation only mask.
- `Sample.MCD64A1.hdf`: Sample MODIS BA data for data structure reference.
- `fVAF_Defo_2013-2022.nc`: Climatological mean deforestation fraction for active fires.
- `CFmean_2014-2023.nc`: Climatological mean cloud fraction.
- `fraction_forest_BA.txt`: 0.25 deg forest fraction.
- `GFEDregions_025d.tif`: 0.25 deg GFED region mask.
- `mjLCT_MCD12C1_0.25x0.25.hdf`: 0.25 deg major land cover type data.
- `MCD12Q1_LCd_025d_2022.tif`: 0.25 deg GFED5 land cover data for 2022.
- `VNP14IMG/`, `VNP14IMGDL/`, `VNP14IMG_NRT/`: VIIRS active fire data (standard, daily, NRT).
- `MOD500mLatLon/`: MODIS 500m latitude and longitude data.
- `MaskPeat/`: 500m Peatland mask.
- `MaskDef_2022/`: 500m Deforestation mask for 2022.
- `MCD12Q1_LCd_2022/`: 500m MODIS land cover data for 2022.

## Key Outputs

- `Intermediate/VAF500m/`: Daily VAF data at 500m resolution.
- `Intermediate/VAF/`: Daily VAF data at 0.25 degree resolution.
- `Intermediate/BA/`: GFED5e NRT Burned Area (BA) data.
- `Intermediate/EM/`: GFED5e NRT Emissions (EM) data.
- `Output/<year>/GFED5eNRTeco_<date>.nc`: Combined 16-class data (VAF + BA + EM).

## Burning Type Classification

The script uses four burning type classification schemes:
- **Modified MODIS LCT (MOD)**
- **GFED5 BA LCT (GBA)**
- **GFED5 16-class LCT (G16)**
- **GFED5 6-class LCT (G6)**

These are applied differently for EFA, FC, VFA, BA, and EM calculations.

## Main Steps (Workflow)

The script follows a structured workflow:

- **Step 0: VIIRS data download and pre-processing**
    - Reads Earthdata token.
    - Fetches and downloads VIIRS active fire data (NRT or standard) using `wget`.
    - Converts raw NetCDF data to daily fire location (DL) CSV files.
    - Key functions: `read_earthdata_token`, `get_remote_ts`, `py_wget`, `download_VNP14IMG_daily`, `convert_VNP14IMG_to_DL`, `make_VNP14IMGDL`.

- **Step 1: Record VAF at 500m resolution**
    - Reads and preprocesses VIIRS daily active fire location data.
    - Filters out static pixels and adds major land cover type and day/night flags.
    - Records VAF data to MODIS-500m grid cells for each tile.
    - Key functions: `readpreprocess_DL`, `removestaticpixels`, `adddfmjLCT`, `add_wgt_2_VNP`, `cal_VAF_1tile_1day`, `recordVAF500m`.

- **Step 2: Record VAF sums (GBA groups format) at 0.25deg resolution**
    - Converts 2D VAF maps to 1D arrays of counts in each 0.25 deg grid cell.
    - Calculates and records VAF for all types and biomes, including peat and deforestation masks.
    - Adjusts VAF based on deforestation climatology.
    - Saves the processed VAF data to NetCDF files.
    - Key functions: `cal_VAFp25_1tile_1day`, `cal_VAFp25_alltiles_1day`, `ds_VAF_1day_reformat`, `doVAFadjust_daily`.

- **Step 3: Use pre-calculated scalar to convert VAF sums to BA sums (G16 format)**
    - Reads GFED regional mask and daily VIIRS AF data.
    - Applies EFA scalars to convert VAF to BA for different biomes and types.
    - Converts the BA data to a 16-class format.
    - Saves the BA data to NetCDF files.
    - Key functions: `read_GFEDmask`, `read_VFA`, `mapGFEDregdata`, `cal_BA_scled_day`, `remapBAclass`.

- **Step 4: Use pre-calculated scalar to convert BA sums to EM sums (G16 format)**
    - Reads the 16-class BA data.
    - Applies Fuel Consumption (FC) scalars to convert BA to EM.
    - Saves the EM data to NetCDF files.
    - Key functions: `read_FC`, `cal_EM_scled_day`.

- **Step 5: Derive GFED5eNRTeco (16-class combined data VAF + BA + EM)**
    - Combines VAF, BA, and EM data into a single 16-class dataset.
    - Adds global and variable-specific attributes to the combined dataset.
    - Key functions: `getVAF16class`, `make_GFED5eco`, `add_GFED5eco_attrs`, `add_BA_attrs`, `add_EM_attrs`, `add_VAF_attrs`.

## Constants

- `G25_lats`, `G25_lons`: Latitude and longitude arrays for the 0.25-degree grid.
- `GFEDnms`: List of GFED region names.
- `LCnms`, `LCnms_full`: Land cover names for GFED5.1 burned area.
- `EMLCnms`: 16-class Land cover names for GFED5.1 emissions.

## Running Instructions

1.  **Set up Python environment:** Ensure the following external Python modules are installed: `pyhdf`, `gdal`, `xarray`, `numpy`, `pandas`, `pyproj`, `pynio`, `scipy`.
2.  **Set running directory:** Modify the `dirData` variable in the script to your desired running directory (default is `~/GFED5eNRT/`).
3.  **Set NASA Earthdata credential:** Copy your Earthdata token into `Earthdata_token.txt` within the `Input` subdirectory.
4.  **Run the code:** Execute the Python script.

## Utility Functions

The script includes several utility functions for common tasks:

- `strymd`, `strdoy`: Date formatting.
- `mkdir`: Directory creation.
- `to_netcdf`: Saving xarray Datasets to NetCDF with compression.
- `nowarn`: Suppressing Python warnings.
- `read_BA`, `read_EM`, `read_VNP14IMG_NRT_daily`, `read_VNP14IMGML_daily`, `read_GFED5eco`: Functions for reading intermediate and output data.
- `MODtilegt`, `prj4sinus`, `sinusproj`, `FCpoints2arr`, `getMODlatlon`, `getMODlatlon_hdf`, `get_tile_paras`, `set_FCtile_ds`: Functions for processing VIIRS data at 500m MODIS sinusoidal grid cells.
- `tif2arr`, `getMODLC`, `D2dcoarser`, `D2dfiner`, `getDEFM`, `getPEATM`: Functions for reading 500m ancillary raster data.
