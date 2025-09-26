#!/usr/bin/env python3
"""
This file contains configuration details for 
- data directory
- credentials
- connecting to the FTP server.
It is designed to be imported by other Python scripts.
"""
import os

# ------------------
# Data directory
# ------------------
dirData = os.path.expanduser('~/GFED5eNRT/')

# ------------------
# Credentials
# ------------------
# For security, add the following environment variables in your .bashrc
# export EARTHDATA_PAT="earthdata_token"

# ------------------
# Ftp site - UCI
# ------------------
# For security, add the following environment variables in your .bashrc
# export SFTP_UCI_UN="ftp_user_name"
# export SFTP_UCI_PW="ftp_password"

# The hostname or IP address of the FTP server.
ftpurl_UCI = 'home.ps.uci.edu'

# The username for the FTP connection, retrieved from an environment variable.
# Using environment variables is a secure way to handle credentials.
ftpun_UCI = os.environ.get("SFTP_UCI_UN")

# The password for the FTP connection, also from an environment variable.
ftppw_UCI = os.environ.get("SFTP_UCI_PW")

# The port number for the FTP service. Port 22 is for SFTP.
ftpport_UCI = 22

# The specific directory on the server to access.
ftpdir_UCI = "/home/users/ess/ychen17/public_html/Shared/GFED5eNRT/"

# ------------------
# Ftp site - WUR
# ------------------
# For security, add the following environment variables in your .bashrc
# export SFTP_WUR_UN="ftp_user_name"
# export SFTP_WUR_PW="ftp_password"

# The hostname or IP address of the FTP server.
ftpurl_WUR = 'ftp.prd.dip.wur.nl'

# The username for the FTP connection, retrieved from an environment variable.
# Using environment variables is a secure way to handle credentials.
ftpun_WUR = os.environ.get("SFTP_WUR_UN")

# The password for the FTP connection, also from an environment variable.
ftppw_WUR = os.environ.get("SFTP_WUR_PW")

# The port number for the FTP service. Port 22 is for SFTP.
ftpport_WUR = 1022

# The specific directory on the server to access.
ftpdir_WUR = "/GFED5/GFED5.1ext_NRT_Beta/"