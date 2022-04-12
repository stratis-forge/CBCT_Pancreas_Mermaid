# CBCT_Pancreas_Mermaid

##Description

Deep-learning (DL) registration model to predict OAR segmentations on the CBCT derived from segmentations on the planning CT.

## Code Location

Source code is hosted on the public-facing Stratis-Forge (https://github.com/stratis-forge)

## Software Verification via Test Dataset
### Generated Output
* RTSTRUCT with warped structures to match input CBCT
* DICOM with cropped FOV CBCT
* HPC Implementation

## Image Input Requirements

To run the routine, the process requires the following images in DICOM format:

* planning CT 
* RTSTRUCT file with the following contours (drawn on planning CT):
* Bowel_Sm  (Small bowel)
* Stomach_duo  (Stomach duodenum)
* LUNG_L 
* LUNG_R (or optionally, combined "LUNGS" structure) 
* ISOCENTER  (point/fiducial)
* ROI  (15mm-enlarged PTV4500)
* Conebeam CT (CBCT), rigidly aligned to isocenter and resliced to match planning CT

Forthcoming change: code will have ability to work from single, combined "LUNGS" structure. Some flexible/dynamic determination of these structure names.

## Input Folder Format
### INPUTDIR
* CT1.3.46.670589.33.1.63754445356596240200002.525470822017xxxxx   (planning CT)
* OnlineMatchResliced  (resliced CBCT)
* RTSStandardized (RTSTRUCT on planning CT)

## Run Script

```
#!/bin/bash

INPUTDIR=${1?Error: no input dicom dir}
AWAREOUT=${2?Error: no output dicom dir}

#EMAILDIST="aptea@mskcc.org,huj@mskcc.org,locastre@mskcc.org,magerasg@mskcc.org"
EMAILDIST="huj@mskcc.org,locastre@mskcc.org"
SERVERLIST="pllimphsing1 pllimphsing2 pllimphsing3 pllimphsing4"
log_dir=/cluster/home/clinSegData/logs/eclipse_test
LOGPFX=mermaid
ptname=`basename $INPUTDIR`


{
bsub -m "${SERVERLIST}" \
     -N -u "${EMAILDIST}" \
     -o $log_dir/${LOGPFX}output.pt$ptname.%J \
     -e $log_dir/${LOGPFX}errors.pt$ptname.%J \
     -R rusage[mem=25GB] -q clinical -n 1 \
     -W 00:20  -gpu "num=1:mode=exclusive_process:mps=no:j_exclusive=yes" \
     -I /cluster/home/clinSegData/scripts/eclipse_test/mermaid_process.sh ${INPUTDIR} ${AWAREOUT}
} || {
            echo failed
        echo "======== $1 job failed =========" | \
        mail -s "MP_CLIN1 JOB FAILURE ${LOGPFX}" "${EMAILDIST}"
}
```


## References
* Han X, Hong J, Reyngold M, Crane C, Cuaron J, Hajj C, Mann J, Zinovoy M, Greer H, Yorke E, Mageras G, Niethammer M. Deep-learning-based image registration and automatic segmentation of organs-at-risk in cone-beam CT scans from high-dose radiation treatment of pancreatic cancer. Med Phys. 2021 Jun;48(6):3084-3095. doi: 10.1002/mp.14906. Epub 2021 May 14. PMID: 33905539.
