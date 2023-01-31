#!/bin/bash

# Home Directory to source .bashrc file
FILE=$HOME/.bashrc
if [ -f "$FILE" ]; then
    source "$FILE"
else 
    exit 1;
fi

# Date
d=$(date +"%Y_%m_%d")

# Define variables
flock=$1
comm=$2
obst_file=$3
time_to_run=$4
obst_type=$(echo ${obst_file##*/} | cut -d"." -f1)
folder_name="${d}_${obst_type}_${flock}_${comm}"

#Create associated folder
cd $HOME/optim
mkdir -p robotsim-master/parameters/flocking_folder/$folder_name
mkdir -p robotsim-master/output_folder/$folder_name

# Compile source code in server mode
cd $HOME/optim/robotsim-master
make optim server=true

# Run NSGA3 script
cd $HOME/optim
python3 NSGAIII_pymoo_press.py $flock $comm $obst_file $time_to_run $d

