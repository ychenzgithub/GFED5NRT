import GFED5NRT 

if __name__ == '__main__':
    # for mo in range(1,12+1):
    #     GFED5NRT.convert2mon_all(2025,mo)

    # yr = 2023  # reformat to annual ecosystem files
    # GFED5NRT.reformat_GFED5NRT_eco(yr,sat='CMB')

    # generate time-series figures for a specific date
    fnmdaily, fnmcumu = GFED5NRT.generatetsfig_fordate(2025, 10, 12, vnm='EM', sat='CMB')
