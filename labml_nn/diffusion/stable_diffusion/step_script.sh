#!/bin/bash

# Define the step sizes to use
STEPS=(10 30 50 100 200 500 1000)

mkdir ddpmOuts
cd ddpmOuts
# Loop over the step sizes and run the command
for STEP in "${STEPS[@]}"
do
    mkdir $STEP
    cd $STEP
    python3 ../../scripts/text_to_image.py --sampler ddpm --steps "$STEP" --prompt "a squirrel swimming freestyle in a swimming pool" --batch_size 2
    cd ..
done
