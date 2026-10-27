using Pkg
#using WoundHealingAnalysis
using Images
using FileIO
using Plots

# Locate the project independently of the working directory.
const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_DIR,"src","load_segmentation_hyperparams.jl"))
include(joinpath(PROJECT_DIR,"src","segment_image.jl"))

# Choose one representative image.
image_path = joinpath(PROJECT_DIR, "examples/")
image_filename = "example_image.tif"


steps = load_preprocessing_pipeline(joinpath(PROJECT_DIR,"configs","PreprocessingSteps.toml"))
segmentation_hyperparams = load_data_from_hyperparams_file(joinpath(PROJECT_DIR,"configs","HyperParams.toml"))



segments, labels_segments, segmented_final_im = segmentation_steps(image_path, image_filename, segmentation_hyperparams; display_steps = true,)

#****** esto lo debería hacer en segmentation steps, idem los hyperparams para hacer seguimiento !!
save_preprocessing_pipeline(steps, joinpath(PROJECT_DIR,"examples","PreprocessingSteps.toml"))

