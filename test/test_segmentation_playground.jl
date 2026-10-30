
#using WoundHealingAnalysis

project_dir = normpath(joinpath(@__DIR__, ".."))

include(joinpath(project_dir,"src","load_segmentation_hyperparams.jl"))
include(joinpath(project_dir,"src","segment_image.jl"))


# Select hyperparameters and preprocessing configurations for segmentation
hyperparams_file = joinpath(project_dir, "configs", "Hyperparams.toml")
preprocessing_file = joinpath(project_dir, "configs", "PreprocessingSteps.toml")


# Choose one representative image.
image_dir = joinpath(project_dir, "examples/")
image_filename = "example_image.tif"
image_path = joinpath(image_dir, image_filename)

# Load selected hyperparameters and preprocessing configurations for segmentation
steps = load_preprocessing_pipeline(preprocessing_file)
segmentation_hyperparams = load_data_from_hyperparams_file(hyperparams_file)


# Run segmentation
segments, labels_segments, segmented_final_im = segmentation_steps(load(image_path), image_path, segmentation_hyperparams; steps=steps, display_steps = true,)


save_preprocessing_pipeline(steps, joinpath(project_dir,"examples","PreprocessingSteps.toml"))
save_hyperparams_to_file(segmentation_hyperparams, joinpath(project_dir,"examples","Hyperparams.toml"))

