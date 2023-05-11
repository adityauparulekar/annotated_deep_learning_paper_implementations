#!/bin/bash

# Define the step sizes to use
STEPS=(5 10 15 20 25 30 35 40 45 50)

# Loop over the step sizes and run the command
for STEP in "${STEPS[@]}"
do
    mkdir $STEP
    cd $STEP
    python3 ../scripts/text_to_image.py --sampler ddim --steps "$STEP" --prompt "an astronaut at an ice cream shop"
    cd ..
done
