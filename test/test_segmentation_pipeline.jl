#using WoundHealingAnalysis

project_dir = normpath(joinpath(@__DIR__, ".."))

include(joinpath(project_dir,"src","segmentation_pipeline.jl"))
include(joinpath(project_dir,"src","load_data.jl"))

config_file = joinpath(project_dir,"examples","example_experiment1","config_exp1.toml")
hyperparams_file = joinpath(project_dir, "configs", "Hyperparams.toml")
preprocessing_steps_file = joinpath(project_dir, "configs", "PreprocessingSteps.toml")

data = load_data_from_config_file(config_file)
segmentation_hyperparams = load_data_from_hyperparams_file(hyperparams_file)
preprocessing_steps = load_preprocessing_pipeline(preprocessing_steps_file)

segment_all_files(data, segmentation_hyperparams, preprocessing_steps)

save_preprocessing_pipeline(preprocessing_steps, joinpath(project_dir,"examples","example_experiment1","PreprocessingSteps.toml"))
save_hyperparams_to_file(segmentation_hyperparams, joinpath(project_dir,"examples","example_experiment1","Hyperparams.toml"))
