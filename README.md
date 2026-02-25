# GFED5 NRT Extension (GFED5NRT)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The **GFED5 NRT Extension (GFED5NRT)** is a professional-grade scientific tool designed to calculate Near Real-Time (NRT) global burned area (BA) and fire emissions (EM) at a 0.25-degree spatial resolution. 

Building upon the Global Fire Emissions Database (GFED5), this system leverages **VIIRS 375m active fire data** from the Suomi NPP and NOAA-20 (JPSS-1) satellites. It employs a sophisticated two-step scaling methodology:
1.  **BA Derivation:** Daily VIIRS active fire counts (VAF) are converted to burned area using pre-derived **Effective Fire Area (EFA)** scalars, categorized by region and land cover type.
2.  **EM Derivation:** The calculated burned area is combined with pre-calculated **Fuel Consumption (FC)** scalars and species-specific **Emission Factors (EF)** to estimate chemical emissions.

The system automates the entire pipeline from data acquisition and preprocessing to visualization and cloud synchronization.


## Data Products

The output is categorized by the VIIRS sensors used for derivation and partitioned into two primary data streams:

### Satellite Configurations
*   **CMB (Combined):** Derived from both Suomi NPP and NOAA-20 VIIRS observations.
*   **VNP:** Derived exclusively from Suomi NPP (SNPP) VIIRS data.
*   **VJ1:** Derived exclusively from NOAA-20 (JPSS-1) VIIRS data.

### Data Streams
*   **GFED5NRTeco (Ecosystem):** Contains VAF, BA, and EM partitioned by 16 fire types (e.g., boreal forest, temperate grassland, tropical peat, deforestation).
*   **GFED5NRTspe (Species):** Contains gridded daily and monthly emissions for carbon (C) and essential trace gases/aerosols (CO, CH4, NOx, PM2.5, BC, OC, etc.).


## Directory Structure

```text
GFED5NRT/
├── Code/                   # Source code and configuration
│   ├── GFED5NRT.py         # Core engine for BA and EM calculations
│   ├── userconfig.py       # User-specific paths and credentials
│   ├── GFED5NRT.sh         # Batch execution shell script
│   ├── requirements.txt    # Python dependencies (pip)
│   └── environment.yml     # Conda environment definition
├── Input/                  # Critical auxiliary data (masks, scalars, EF tables)
├── Intermediate/           # Temporary processing files (500m VAF tiles)
└── Output/                 # Final NetCDF products
```


## Installation & Setup

### 1. Environment Configuration
Ensure you have a Python 3.9+ environment. You can install the required dependencies using either Conda or Pip:

**Using Conda:**
```bash
conda env create -f Code/environment.yml
conda activate gfed5enrt
```

**Using Pip:**
```bash
pip install -r Code/requirements.txt
```

### 2. User Configuration
Update the `Code/userconfig.py` file with your specific environment details:
*   Project root directory (`dirData`).
*   Earthdata Login credentials (stored via environment variables like `EARTHDATA_PAT`). See `3. Authentication` for details.
*   Optional: SFTP server details for data synchronization.

### 3. Authentication
The code requires NASA Earthdata credentials to download VIIRS data. For security, export your token as an environment variable in your `~/.bashrc`:
```bash
export EARTHDATA_PAT="your_token_here"
```

### 4. Prepare input data

The `Input/` directory contains various datasets crucial for the calculations:
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
- `table_EM_2002-2022.csv`: global and regional monthly sum of GFED5 EM for 2002-2022
- `VNP14IMG/`, `VNP14IMGDL/`, `VNP14IMG_NRT/`: VIIRS active fire data (standard, daily, NRT).
- `MOD500mLatLon/`: MODIS 500m latitude and longitude data.
- `MaskPeat/`: 500m Peatland mask.
- `MaskDef_2022/`: 500m Deforestation mask for 2022.
- `MCD12Q1_LCd_2022/`: 500m MODIS land cover data for 2022.


## Usage

### Manual Execution
To run the main processing pipeline for a specific date range or search for pending updates:
```bash
python Code/GFED5NRT.py
```

### Automated Processing (Cron)
The system is designed for operational use. You can schedule the `GFED5NRT.sh` script via Cron. Note that Cron environments are minimal; ensure you source your profile in the script:
```bash
# Example Cron job (runs daily at 7:00 AM)
0 7 * * * /bin/sh /path/to/GFED5NRT/Code/GFED5NRT.sh >> /path/to/GFED5NRT/Code/cron.log 2>&1
```


## Technical Workflow

The system operates in six distinct phases:

1.  **Ingestion (Step 0):** Automated download of VIIRS active fire data (NRT/Standard) from NASA Earthdata and conversion to daily fire location CSVs.
2.  **Mapping (Step 1):** Alignment of fire detections to the MODIS 500m Sinusoidal grid, filtering static sources (gas flares, etc.), and assigning land cover types.
3.  **Aggregation (Step 2):** Resampling of 500m detections to 0.25° grid cells, incorporating peat and deforestation climatologies.
4.  **BA Calculation (Step 3):** Application of **EFA scalars** to convert VAF counts into burned area units across 16 land cover classes.
5.  **EM Calculation (Step 4):** Conversion of BA to mass emissions using **Fuel Consumption (FC)** scalars.
6.  **Product Generation (Step 5):** Consolidation into final NetCDF files (Ecosystem and Species streams) with full metadata and archival-quality compression.



## Data

Routinely updated GFED5NRT products, including the reprocessed 2023–2024 data, are freely accessible at [www.globalfiredata.org](https://www.globalfiredata.org/). 



## References & Citations

If you use this dataset or code, please cite the following:

*   **Chen et al.** (submitted). *Tracking recent extremes and interannual variability of global fire emissions using a near-real-time extension to the Global Fire Emissions Database.*
*   **van der Werf et al.** (2025). *Landscape fire emissions from the 5th version of the Global Fire Emissions Database (GFED5).* Scientific Data.



## Contact

**Yang Chen**  
Department of Earth System Science  
University of California, Irvine  
Email: [yang.chen@uci.edu](mailto:yang.chen@uci.edu)