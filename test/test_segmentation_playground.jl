
#using WoundHealingAnalysis

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))

include(joinpath(PROJECT_DIR,"src","load_segmentation_hyperparams.jl"))
include(joinpath(PROJECT_DIR,"src","segment_image.jl"))


# Select hyperparameters and preprocessing configurations for segmentation
const HYPERPARAMS_FILE = joinpath(PROJECT_DIR, "configs", "Hyperparams.toml")
const PREPROCESSING_FILE = joinpath(PROJECT_DIR, "configs", "PreprocessingSteps.toml")


# Choose one representative image.
image_dir = joinpath(PROJECT_DIR, "examples/")
image_filename = "example_image.tif"


# Load selected hyperparameters and preprocessing configurations for segmentation
steps = load_preprocessing_pipeline(PREPROCESSING_FILE)
segmentation_hyperparams = load_data_from_hyperparams_file(HYPERPARAMS_FILE)


# Run segmentation
segments, labels_segments, segmented_final_im = segmentation_steps(image_dir, image_filename, segmentation_hyperparams; steps=steps, display_steps = true,)


save_preprocessing_pipeline(steps, joinpath(PROJECT_DIR,"examples","PreprocessingSteps.toml"))
save_hyperparams_to_file(segmentation_hyperparams, joinpath(PROJECT_DIR,"examples","Hyperparams.toml"))

