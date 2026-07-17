#using Pkg
#Pkg.activate(joinpath(@__DIR__, ".."))

#include(joinpath(@__DIR__, "..", "src", "WoundHealingAnalysis.jl"))


#using .WoundHealingAnalysis

main_code_folder = @__DIR__

println(joinpath(main_code_folder,"..","src","load_data.jl"))
include(joinpath(main_code_folder,"..","src","load_data.jl"))

#code_folder = "c:/Users/Usuario/Downloads/segmentation_code/"
project_dir = joinpath(main_code_folder, "..","config")
config_file = joinpath(project_dir,"config.toml")

data = load_data_from_config_file(config_file)

