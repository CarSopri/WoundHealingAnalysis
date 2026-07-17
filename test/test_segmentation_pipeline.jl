#using WoundHealingAnalysis

main_code_folder = @__DIR__

include(joinpath(main_code_folder,"..","src","segmentation_pipeline.jl"))
include(joinpath(main_code_folder,"..","src","load_data.jl"))

config_file = joinpath(main_code_folder,"..","config","config-prueba.toml")

data = load_data_from_config_file(config_file)

segmentation_hyperparams = SegmentationHyperparameters()

run_segmentation_for_all_files(data, segmentation_hyperparams)
