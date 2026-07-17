# Segmentation pipeline for wound closure analysis
All images tested came from the Cytation 10 microscope.

Complete the config.toml file with your file information.
Then run in the terminal inside the code directory:
julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

Run these in the terminal:
- test_load_data.jl --> If you want to check that the config.toml file has all the information loaded correctly
- test_segmentation_pipeline.jl --> If you want to segment the images as indicated in the config.toml file
- test_segmentation_analysis_pipeline.jl --> If you want the table with the time taken between frames (time_report.txt) and the table with the wound areas (in px) for every image (scratch_report.txt).
