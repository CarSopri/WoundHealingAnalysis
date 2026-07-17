#using WoundHealingAnalysis

using JLD2

main_code_folder = @__DIR__
include(joinpath(main_code_folder,"..","src","construct_walls.jl"))
include(joinpath(main_code_folder,"..","src","load_data.jl"))

#**bueno ojo en esta parte con lo que pueda ser distinto de fibroblastos 1 con el resto de experimentos !!!
#***also revisar cómo son distintos exclude e ignore_pits !! creo que exclude ya incluye los ignore_pits detectados anteriormente no? o sea que lo puse a mano en realidad para ignorarlos de antemano, pero capaz debería ser un .txt que sirva para 
# no tener que volver a perder el tiempo corriendo cosas que ya vi que no andan, sin tener que cargar a mano cuáles son


#**ThreadsX para paralelizar


# Plotting setup
#default(palette = palette(:Dark2_8))
#figs = [plot(layout=(1,2), size=(800,400), legend=true, grid=true, margin=5mm) for _ in 1:6]
# figsize=(25,6)

project_dir = joinpath(main_code_folder, "..","config")
config_file = joinpath(project_dir,"config-prueba.toml")

data = load_data_from_config_file(config_file)
data.ignore_pits = ignore_rois_from_this_int(data.keep_rois, [l for l in data.pit_names],length(data.rois))

seg_data = analyze_segmentation(data)
tf = seg_data.tf

#**** esto que sigue debería estar en un test_timelapse_analysis ?
pits_old = setdiff(data.pit_names, seg_data.ignore_pits)

#for d in unique(seg_data.ignore_pits)
#    @info (d,", ",count(i->(i== d), seg_data.ignore_pits)," ; ")
#end
@info ("\n")
for d in unique(seg_data.failed_segmentations)
    @info (d,", ",count(i->(i== d), seg_data.failed_segmentations)," ; ")
end
@info ("\n")
@info ("Ignored pits: ",length(unique(seg_data.ignore_pits)))
@info ("Frame threshold to consider sequence valid: if more than ",tf*0.6," frames failed.") #**que ese 75% también sea un parámetro de análisis
@info ("Failed segmentations (not enough valid frames): ",length(unique([d for d in seg_data.failed_segmentations if count(i->(i== d), seg_data.failed_segmentations)>(tf*0.75)])))
pits_old = setdiff([p[1:2]*"ROI"*p[end] for p in data.all_files], seg_data.ignore_pits) #***esto hay que corregirlo porque all_files tiene las combinaciones de frame también en el nombre we
pits_old = setdiff(pits_old, [d for d in seg_data.failed_segmentations if count(i->(i== d), seg_data.failed_segmentations)>(tf*0.75)])
@info ("Ignored pits + any sequence with at least one failed segmentation: ",length(unique(vcat(seg_data.ignore_pits, seg_data.failed_segmentations))))
@info ("Complete segmentations: ",length(pits_old))

#***adentro del test, printear cómo salieron las variables del jld2 para chequear que los tamaños estén bien idk
#println(widths_distribution)
#println(left_walls)
#println(right_walls)
#print(left_roughnesses)
#print(right_roughnesses)
#right_walls = Dict{String, Vector{Float64}}()
#left_roughnesses = Dict{String, Vector{Float64}}()
#right_roughnesses = Dict{String, Vector{Float64}}()

@save data.main_folder*"DataFromSegmentation.jld2" seg_data


