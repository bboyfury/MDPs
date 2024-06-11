
# Simple Grid & Tiger Problem POMDP

## Overview
This module simulates the 1x4 grid and the "Tiger Problem" as a POMDP using `pomdp_py`. It includes state, action, and observation definitions, models for transitions, observations, and rewards, and a policy for decision-making. Belief states are visualized using `matplotlib`.

## Tiger Problem default values:

discount factor=0.95 
Horizon=3
Tstep=4
---
## Grid:
num_actions = 5
action='EAST'
or
action='WEST'

## Grid Setup and Running:
```bash
python SimpleGrid.py
```

## Tiger Setup
Ensure `pomdp_py` and `matplotlib` are installed:
```bash
pip install pomdp-py matplotlib
```

## Running the Simulation

Execute the script to run the simulation:
```bash
python Tiger.py
```
Execute the script to run the simulation and visualize the belief updates:
```bash
python TigerDraw.py
```

## Visualization
Belief states are visualized after each action to show probabilities of the tiger's positions.

