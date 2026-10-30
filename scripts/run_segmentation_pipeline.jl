

function run_segmentation_pipeline(config_file::String, hyperparams_file::String, preprocessing_steps_file::String)
    data = load_data_from_config_file(config_file)
    segmentation_hyperparams = load_data_from_hyperparams_file(hyperparams_file)
    preprocessing_steps = load_preprocessing_pipeline(preprocessing_steps_file)

    segment_all_files(data, segmentation_hyperparams, preprocessing_steps)

    save_preprocessing_pipeline(preprocessing_steps, joinpath(data.main_folder, "PreprocessingSteps.toml"))
    save_hyperparams_to_file(segmentation_hyperparams, joinpath(data.main_folder, "Hyperparams.toml"))
end
