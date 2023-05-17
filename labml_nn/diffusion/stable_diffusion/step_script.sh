#!/bin/bash

# Define the step sizes to use
STEPS=(0.49 0.23 0.147 0.077 0.0395 0.01602 0.00805)

mkdir ourOuts
cd ourOuts
# Loop over the step sizes and run the command
for STEP in "${STEPS[@]}"
do
    mkdir $STEP
    cd $STEP
    python3 ../../scripts/text_to_image.py --sampler ddpm --step_size_eps "$STEP" --prompt "a squirrel swimming freestyle in a swimming pool" --batch_size 2
    cd ..
done
