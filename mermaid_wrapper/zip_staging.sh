#!/bin/bash

#File: /software/zip_staging.sh
#Use:  Organizes DICOM's received from AWARE and positions them for pre-processing

SUBJ="$1"
CTDIR="$2"
CBDIR="$3"
RTDIR="$4"

ZIPSTAGING=/scratch/tmp
DICOMDIR=/scratch/dicoms

cd $ZIPSTAGING

mkdir planCT_OG CBCT01_OG
mv ${CTDIR} CT_1900
mv ${RTDIR} RTSStandardized
mv {CT_1900,RTSStandardized} planCT_OG
mv ${CBDIR} CT_OnlineMatchResliced
mv CT_OnlineMatchResliced CBCT01_OG

mkdir $SUBJ
mv {planCT_OG,CBCT01_OG} $SUBJ

#zip -r ${SUBJ}.zip ${SUBJ}
#mv ${SUBJ}.zip ../zips

mv ${SUBJ} $DICOMDIR

#rm -rf ${SUBJ}
