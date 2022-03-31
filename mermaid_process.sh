#!/bin/bash

CTPATH=${1?Error: no input dicom dir}
CBPATH=${2?Error: no input dicom dir}
RTPATH=${3?Error: no input dicom dir}
AWAREOUT=${4?Error: no output dicom dir}

CTDIR=`basename ${CTPATH}`
CBDIR=`basename ${CBPATH}`
RTDIR=`basename ${RTPATH}`

INPUTDIR=`dirname ${CTPATH}`
#INPUTDIR=${1?Error: no input dicom dir}
#AWAREOUT=${2?Error: no output dicom dir}
#SESSIONDIR=${3?Error: no session dir}
#SIF=${4?Error: no container path}

## CONTAINER FILE
SIF=/cluster/home/clinSegData/containers/eclipse_test/mermaid/UNC_mermaid_clinical_container.def_b506c.sif

# Get Subject MRN from first DICOM in INPUTDIR
#CT0=`find ${INPUTDIR} -name "*.dcm" | head -1`
ptname=`basename $INPUTDIR`
SUBJ=`basename $INPUTDIR | awk -F_ '{ print $1 }'`
echo Working subject is ${SUBJ}

## SET WORKING PATHS VARIABLES
SESSIONDIR=/cluster/home/clinSegData/sessions/session${ptname}
echo Session Directory is ${SESSIONDIR}
ZIPSTAGING=${SESSIONDIR}/tmp
echo Staging directory is ${ZIPSTAGING}
ZIPDIR=${SESSIONDIR}/zips
echo Staging directory is ${ZIPDIR}
DICOMDIR=${SESSIONDIR}/dicoms
echo DICOM directory is ${DICOMDIR}
TESTDIR=${SESSIONDIR}/test_result_best_eval.pth.tar
echo Path to model output is ${TESTDIR}
OUTDIR=${SESSIONDIR}/output
echo Result output dicom to ${OUTDIR}

#SESSIONDIR is the scratch directory; also create the directory that will mount to the containers as test output
mkdir ${SESSIONDIR}
mkdir ${ZIPSTAGING} ${ZIPDIR} ${DICOMDIR} ${TESTDIR} ${OUTDIR}
cd ${ZIPDIR} && ln -s /software/preprocess/template.zip && cd ${SESSIONDIR}

#Copy intput DICOM data to staging directory
rsync -rvPz ${INPUTDIR}/ ${ZIPSTAGING}

echo "Executing container command: "

echo singularity run --nv --app mermaid --bind ${SESSIONDIR}:/scratch ${SIF} ${SUBJ}
#app to run everything: stage data, preprocess, create data list, run through network, output DICOM
singularity run --nv --app mermaid --bind ${SESSIONDIR}:/scratch ${SIF} ${SUBJ} ${CTDIR} ${CBDIR} ${RTDIR}


#copy output to Jason's location
rsync -rvPz ${OUTDIR}/ ${AWAREOUT}

