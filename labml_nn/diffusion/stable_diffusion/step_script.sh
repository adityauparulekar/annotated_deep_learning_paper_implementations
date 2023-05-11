#!/bin/bash

# Define the step sizes to use
STEPS=(0.7 0.49 0.39 0.32 0.27 0.23 0.2 0.18 0.16 0.148)

# Loop over the step sizes and run the command
for STEP in "${STEPS[@]}"
do
    mkdir $STEP
    cd $STEP
    python3 ../scripts/text_to_image.py --sampler ddpm --step_size_eps "$STEP" --prompt "an astronaut at an ice cream shop"
    cd ..
done
