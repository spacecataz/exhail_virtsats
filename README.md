# exhail_virtsats
Create and analyze virtual satellite fly-throughs of BATS-R-US results for the
ExHAIL Small Explorer (SMEX) proposal.

This repository contains code and sample data for the purposes of testing
the hypothesis that given enough spacecraft, each observing ion outflow
along their trajectories, the global outflow spatial distribution can be
reconstructed.
The two main tools for accomplishing this are the `ExHail.py` python module and
the executable `sat_analysis.py` script. These scripts load BATS-R-US output
as "ground truth" and fly virtual satellites through them to produce example
spacecraft datasets.

Sample data files for testing analysis can be found in the `sample_data`
folder. Publication and analysis-critical scripts for bulk processing can
be found in the `analysis scripts folder`.

## Requirements
Users must install Spacepy and all associated requirements.
Further, for handling legacy SWMF file formats, an additional module called
`gmoutflow.py` is required.

## Code To-Do
Continue to update code to switch from post-processed shell files to SWMF
shell output types (removing need for `gmoutflow.py`).

Update `sat_analysis.py` to handle output files from command line option.