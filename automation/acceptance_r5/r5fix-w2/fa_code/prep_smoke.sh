#!/bin/bash
# Assemble the checkpoint-1 smoke policy: the generated pure-NN file
# under the v0 name, with the smoke weights under the filename the
# actor loads (distilled_v1_weights.npz beside the policy file).
cd ~/workspace/carbon-league || exit 1
mkdir -p smoke
cp distilled_v1.py smoke/distilled_v0.py
cp distilled_v0_weights.npz smoke/distilled_v1_weights.npz
ls -la smoke/
