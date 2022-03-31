#!/bin/bash

#
# basic build script for SIF container
# should be run with admin privliege
#
# EM LoCastro June 2020
#
#

SINGRECIPE="${1}"
TMPDIR="${2}"


if [ -z "${SINGRECIPE}" ]
then
	echo "*****************************************************"
	echo "Usage:"
	echo "	sudo ./build_cmd.sh <path-to-sing-def>"
	echo "*****************************************************"
	exit
fi


if [ -z "${TMPDIR}" ]
then
	TMPDIR="/singularity"
fi


#get repo name for SIF filename
REPOREMOTE=`git config --get remote.origin.url`
REPONAME=`basename -s .git ${REPOREMOTE}`


#get repo commit hash to stamp output SIF filename
IDSTAMP=`git log | grep commit | head -1 | awk '{ print $2 }' | cut -c1-5`

echo ${IDSTAMP} > hash_id.txt

#full SIF filename
SIFOUT="${REPONAME}_`basename ${SINGRECIPE}`_${IDSTAMP}.sif"


#build the container
singularity build --tmpdir ${TMPDIR} ${SIFOUT} ${SINGRECIPE}


#set container file to be executable
chmod 775 ${SIFOUT}
