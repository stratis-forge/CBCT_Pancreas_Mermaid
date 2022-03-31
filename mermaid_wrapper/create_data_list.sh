#!/bin/bash

SUBJ="$1"

TEMPLATE=/software/data_list_template.txt

cat ${TEMPLATE} | sed "s/PATIENT/$SUBJ/g" > /software/data/$SUBJ/data_list.txt
