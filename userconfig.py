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
# export SFTP_UCI_UN="ftp_user_name"
# export SFTP_UCI_PW="ftp_password"
# export EARTHDATA_PAT="earthdata_token"

# ------------------
# Ftp site
# ------------------
# The hostname or IP address of the FTP server.
ftpurl = 'home.ps.uci.edu'

# The username for the FTP connection, retrieved from an environment variable.
# Using environment variables is a secure way to handle credentials.
ftpun = os.environ.get("SFTP_UCI_UN")

# The password for the FTP connection, also from an environment variable.
ftppw = os.environ.get("SFTP_UCI_PW")

# The port number for the FTP service. Port 22 is for SFTP.
ftpport = 22

# The specific directory on the server to access.
ftpdir = "/home/users/ess/ychen17/public_html/Shared/GFED5eNRT/"

