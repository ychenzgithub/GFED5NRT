# GFED5 extension NRT (GFED5NRT) Run Options

This document explains the different running options available in `GFED5NRT.py` for generating Near-Real-Time (NRT) burned area and emissions data.

## 1. Operational Daily/Monthly Run
These are the standard entry points for processing NRT data for specific time periods.

- **Functions**: `run_1day(yr, mo, day, IMG=True, upd=False, sat='VNP')` and `run_1mon(yr, mo, IMG=True, upd=False)`
- **Data Source**: Downloads VIIRS 375m active fire image data (`VNP14IMG`/`VJ114IMG`) or NRT image data (`VNP14IMG_NRT`/`VJ114IMG_NRT`) from NASA servers.
- **Process**:
    1. Downloads/mirrors VIIRS data for the specified date.
    2. Converts raw image data into fire location datasets (`IMGDL`).
    3. Records active fire counts at a 500m MODIS sinusoidal grid.
    4. Aggregates fire counts to a 0.25° global grid.
    5. Applies scaling factors to derive Burned Area (BA) and Emissions (EM).
    6. Generates output files: `GFED5NRTeco` (VAF, BA, EM by land cover) and `GFED5NRTspe` (EM by chemical species).
    7. Optionally uploads the results to the WUR server.
- **Usage**: Used for routine daily operations or processing a specific past month.

---

## 2. Operational "Update-to-Now" Run
A workflow wrapper designed for automation (e.g., cron jobs) to keep the repository and public figures current.

- **Functions**: `update_to_now()` and `update_to_now_figs()`
- **Logic**: 
    1. **Data Update (`update_to_now`)**:
        - Determines the latest date for which GFED5NRT data already exists.
        - Iteratively calls `run_1day` for every missing day between the last processed date and today.
        - Processes both S-NPP (`VNP`) and NOAA-20 (`VJ1`) satellites and merges them into a combined (`CMB`) product.
        - Performs automated cleanup of temporary/intermediate files.
    2. **Figure Generation (`update_to_now_figs`)**:
        - Updates the time series data for the current year based on the new combined data.
        - Generates regional daily and cumulative emissions charts.
        - Automatically uploads the figures to the UCI and WUR servers for public display.
- **Usage**: Ideal for automated daily updates of both data and visualizations.

---

## 3. Remedy Run
Designed for retrospective processing or data "repair."

- **Function**: `run_1day_remedy(yr, mo, day, IMG=True, upd=False, sat='VNP')`
- **Logic**: Similar to `run_1day`, but specifically flagged as a "remedy" run. 
- **Key Difference**: Retrospective runs typically skip the data uploading step to the public server, allowing for internal validation or fixing local archives without affecting public data streams until ready.
- **Usage**: Used when a specific day's data needs to be reprocessed due to errors, missing input data during the initial run, or updates in scaling factors.

---

## 4. Run with Standard VIIRS Data (ST)
A high-throughput option for historical processing using standard monthly products. This mode typically involves a multi-step workflow to generate a finalized combined monthly and annual dataset.

- **Primary Function**: `mo_prun_ST(yr, mo, sat='VNP', max_workers=1)`
- **Data Source**: Uses standard VIIRS monthly active fire location products (`VNP14IMGML` or `VJ114IMGML`) instead of NRT image data.
- **Optimization**: 
    - Processes data in parallel using `concurrent.futures` (via `max_workers`).
    - Since standard products contain an entire month of fire locations, this mode is significantly faster for processing historical months than downloading individual daily images.

### Post-Processing Steps for ST Workflow:
Following the initial ST run, several steps are performed to harmonize and aggregate the data:

1. **Fill Missing Days (`fillin_1mon`)**: 
   - Identifies dates with missing VIIRS observations (referencing the [FIRMS missing data API](https://firms.modaps.eosdis.nasa.gov/api/missing_data/)).
   - Generates data for missing dates by interpolating between the nearest available days with valid observations.
   - Ensures a continuous daily record for the month.

2. **Combine Satellites (`combineVNPVJ1`)**: 
   - Merges the S-NPP (`VNP`) and NOAA-20 (`VJ1`) data streams into a single combined (`CMB`) data stream.
   - Uses a simple average for grid cells where both satellites have data, or utilizes the available satellite if one is missing.

3. **Monthly Aggregation (`convert2mon_all`)**: 
   - Combines daily files into monthly sum files.
   - This is performed for all satellite streams (`VNP`, `VJ1`, `CMB`) and both product types (`eco` and `spe`).

4. **(Optional) Annual Reformatting (`reformat_GFED5NRT_eco`)**: 
   - Collects all monthly files for a given year and concatenates them along the time dimension.
   - Reformats the variables, dimensions, and global attributes to conform to the standard GFED5 ecosystem distribution format (0.25° resolution, specific metadata, and units).

---

## Summary of Key Arguments
- `IMG=True`: Drive the run using VIIRS image data (standard for NRT).
- `upd=True`: Force-update/overwrite existing fire location data.
- `sat`: Specify satellite (`VNP` for NPP, `VJ1` for NOAA-20, or `CMB` for combined).
- `max_workers`: Number of parallel threads for `ST` (Standard) runs.
