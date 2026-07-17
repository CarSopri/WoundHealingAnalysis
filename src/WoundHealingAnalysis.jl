
module WoundHealingAnalysis

#**prob debería separar esto en varios archivos

main_code_folder = @__DIR__

# test_load_data.jl

using TOML

include(joinpath(main_code_folder,"load_data.jl"))
include(joinpath(main_code_folder,"save_data.jl"))


# test_segmentation_pipeline.jl

using DataFrames
using CSV
using Logging

main_code_folder = @__DIR__

include(joinpath(main_code_folder,"segment_image_low_contrast.jl"))

using Images               # https://juliaimages.org/latest/tutorials/quickstart/
using ImageFiltering       # https://juliaimages.org/ImageFiltering.jl/stable/function_reference/#ImageFiltering.imfilter
using OffsetArrays
using Statistics
using ImageSegmentation
using ImageMorphology      # https://juliaimages.org/latest/pkgs/segmentation/
using StatsBase
using Logging
using Plots

include(joinpath(main_code_folder,"image_analyzers.jl"))

# test_segmentation_analysis_pipeline.jl

using CSV
using Plots
using DataFrames
using Measures

include(joinpath(main_code_folder,"construct_walls.jl"))

using Images
using ImageMorphology
using ImageSegmentation
using Statistics
using Logging

include(joinpath(main_code_folder,"timelapse_analyzers.jl"))

using LsqFit, LinearAlgebra, StaticArrays, Random
include(joinpath(main_code_folder,"fast_fitting.jl"))



end
