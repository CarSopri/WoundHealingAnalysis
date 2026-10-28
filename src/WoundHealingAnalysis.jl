
module WoundHealingAnalysis

main_code_folder = @__DIR__

# test_load_data.jl

#using TOML

include(joinpath(main_code_folder,"load_data.jl"))
export load_data_from_config_file, get_pixel_to_microns_conversion

include(joinpath(main_code_folder,"load_segmentation_hyperparams.jl"))
export load_data_from_hyperparams_file, load_preprocessing_pipeline, save_preprocessing_pipeline #***crear un save_hyperparams

# test_segmentation_pipeline.jl

main_code_folder = @__DIR__

include(joinpath(main_code_folder,"segment_image.jl"))

export apply_preprocessing_step!
export segmentation_steps, run_segmentation

include(joinpath(main_code_folder, "..","scripts", "run_segmentation_playground.jl"))

export run_segmentation_playground

#using Images: load
#using ImageFiltering: imfilter
#using OffsetArrays #****
#using Statistics
#using ImageSegmentation: felzenszwalb
#using ImageMorphology #****
#using StatsBase

#include(joinpath(main_code_folder,"image_analyzers.jl"))

# test_segmentation_analysis_pipeline.jl
"""
using Plots
using Measures #***

#include(joinpath(main_code_folder,"save_data.jl"))
#export init_report_file




include(joinpath(main_code_folder,"construct_walls.jl"))

using Images
using ImageMorphology
using ImageSegmentation
using Statistics

include(joinpath(main_code_folder,"timelapse_analyzers.jl"))

#using LsqFit, LinearAlgebra, StaticArrays, Random
#include(joinpath(main_code_folder,"fast_fitting.jl"))

"""

end
