# AUTOMATIC COLLECTION AND UPLOAD
# OF PIV OUTPUT DATA
#
# Uses a text file to keep track of uploaded PIV outputs, zips everything new,
# and uploads that using internet or LTE.
#
# Author: Kylie Corcoran
# Date: 7/5/25
# ----------------------------------------

import os
import subprocess
import glob
import zipfile
import time
from datetime import datetime

## Script Parameters
sys_id = "spec3"
upload_path = "gdrive:PopPi/" # We uploaded to a Google Drive to start
search_delay = 3600 # Seconds [once every hour?]
cwd = os.getcwd()
cwd = os.path.join(cwd,'spec')

## Functions
def zip_directory(folder_path, zip_output_path, name):
    print("[lte upload] zipping files to upload..")
    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for cwd, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(cwd, file)
                # Store relative path inside zip
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)

def main():
    # Look for upload log
    # log_flag = glob.glob("upload_log.txt");
    with open(os.path.join(cwd,'upload_log.txt'),'w') as f:
	    pass

    while True:
        print("[lte upload] searching...")
        # get time
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        zipname = sys_id + "-" + timestamp
        zipname = zipname.replace(" ","-")
        # Search For Valid Directories [exclusions apply]
        all_dir = glob.glob(os.path.join(cwd, 'save_data', '**/'), recursive=True)
        filtered_dir = [
            x for x in all_dir
            if 'PIV_output' in os.path.basename(os.path.normpath(x)) and 'test' not in x.lower()
        ]

        # Load any paths from log file, compare for duplicates
        with open(os.path.join(cwd,'upload_log.txt'), 'r') as log:
            uploaded_paths = set(line.strip() for line in log if line.strip())
        normalized_dir = [os.path.normpath(d) for d in filtered_dir]
        dirs_to_upload = [x for x in normalized_dir if x not in uploaded_paths]

        # Append to log
        with open(os.path.join(cwd,'upload_log.txt'), 'a') as log:
            for dir_path in dirs_to_upload:
                log.write(dir_path + '\n')

        # Only try to upload something there
        if len(dirs_to_upload) > 1:

            # Zip PIV_Output folders for upload (CSV, not images or IMU for now)
            for dir_path in dirs_to_upload:
                zip_name = os.path.basename(os.path.normpath(dir_path)) + '.zip'
                zip_output_path = os.path.join(cwd,'zipped_uploads', zipname+'.zip')

            # Make sure the output folder exists
            os.makedirs(os.path.join(cwd,'zipped_uploads'), exist_ok=True)
            zip_directory(dir_path, zip_output_path, zipname);

            # Actually Upload?
            print("[lte upload] uploading")
            subprocess.Popen(["rclone","copy",zip_output_path,upload_path, "--progress"])
        else:
            print("[lte upload] no new PIV files detected")
        
        # Wait
        print ("[lte upload] waiting")
        time.sleep(search_delay)

        # Clear
        print("[lte upload] cleaning")
        subprocess.Popen(["rm","-f",os.path.join(cwd,"zipped_uploads","*")])

main()