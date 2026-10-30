
module WoundHealingAnalysis

main_code_folder = @__DIR__


# test_load_data.jl

include(joinpath(main_code_folder,"load_data.jl"))
export load_data_from_config_file, get_pixel_to_microns_conversion

include(joinpath(main_code_folder,"load_segmentation_hyperparams.jl"))
export load_data_from_hyperparams_file, load_preprocessing_pipeline, save_preprocessing_pipeline, save_hyperparams_to_file


# test_segmentation_playground.jl

main_code_folder = @__DIR__

include(joinpath(main_code_folder,"segment_image.jl"))

export apply_preprocessing_step!
export segmentation_steps, run_segmentation

include(joinpath(main_code_folder, "..","scripts", "run_segmentation_playground.jl"))

export run_segmentation_playground


# test_segmentation_pipeline.jl

include(joinpath(main_code_folder, "segmentation_pipeline.jl"))

export segment_all_files

include(joinpath(main_code_folder, "..","scripts", "run_segmentation_pipeline.jl"))

export run_segmentation_pipeline


end
