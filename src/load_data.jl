
using TOML

Base.@kwdef mutable struct DataInfo

    main_folder::String = @__DIR__
    sub_folders::Vector{String} = [""]
            
    file_format::String = ".tif"

    # The number of elements in sub_folders has to be the same as in time_batches and name_time_batches
    time_batches::Vector{Vector{Int}} = []
    name_time_batches::Vector{String} = []
    first_timeframe::String = "001"

    common_name::String = ""
    follow_convention::Vector{String} = ["common_name","rois","name_time_batches","pit_names"]
    conditions::Vector{String} = ["control"]

    row_names::Union{String,Vector{String}} = ""
    col_names::Union{String,Vector{String}} = ""
    pit_conditions::Vector{Int} = []

    pit_names::Vector{String} = []
    ignore_pits::Vector{String} = []
    pits_used::Vector{String} = []
    keep_rois::Vector{Int} = [] #***acomodar esto, el de arriba saca pits directamente y este deja rois. also no sé si esto anda si está vacío ?
    exclude::Vector{String} = []

    rois::Union{String,Vector{String}} = ""

    all_files::Vector{String} = []

    report::Bool = false
    save_images::Bool = false
end

function ignore_rois_from_this_int(cutoff::Int, pit::String, n_rois::Int) #**ojo que este anda solamente para la segunda convención we
    return [pit*"ROI"*string(i) for i in cutoff+1:n_rois]
end
function ignore_rois_from_this_int(list_cutoffs,list_pits,n_rois::Int)
    return reduce(vcat, [ignore_rois_from_this_int(cutoff,pit,n_rois) for (cutoff,pit) in zip(list_cutoffs,list_pits)])
end

#data.keep_rois = ignore_rois_from_this_int(d["which_files"]["keep_rois"],
#                                            [l for l in data.pit_names],length(data.rois)
#                                            )

function load_data_from_config_file(config_file)

    d = TOML.parsefile(config_file)
    data = DataInfo()

    data.main_folder = d["folders"]["main_folder"]
    data.sub_folders = d["folders"]["sub_folders"]
            
    data.file_format = d["format"]["file_format"]

    # The number of elements in sub_folders has to be the same as in time_batches and name_time_batches
    data.time_batches = d["timeframe_name_convention"]["time_batches"]
    data.name_time_batches = d["timeframe_name_convention"]["name_time_batches"]
    data.first_timeframe = d["timeframe_name_convention"]["first_timeframe"]

    data.common_name = d["pit_name_convention"]["common_name"]
    follow_convention = split(d["pit_name_convention"]["follow_convention"],"*")
    #follow_convention = "common_name*roi*name_time_batch*pit_name"
    #follow_convention = "common_name*pit_name*time_batch" ? y roi puede ser 0 en todo caso

    data.conditions = d["pit_name_convention"]["conditions"]

    if d["pit_name_convention"]["naming_mode"] == "manual"
        data.row_names = d["pit_name_convention"]["manual"]["row"]
        data.col_names = d["pit_name_convention"]["manual"]["col"]
        data.pit_names = d["pit_name_convention"]["manual"]["pit_names"]
        data.pit_conditions = d["pit_name_convention"]["manual"]["pit_conditions"]
    elseif d["pit_name_convention"]["naming_mode"] == "generated"
        #data.pit_names = [l*n*"ROI"*string(roi) for l in d["row"] for n in d["col"] for roi in 1:parse(Int64,d["max_rois"])]
        data.row_names = d["pit_name_convention"]["generated"]["row"]
        data.col_names = d["pit_name_convention"]["generated"]["col"]
        data.pit_names = [l*n for l in data.row_names for n in data.col_names]
        data.pit_conditions = d["pit_name_convention"]["generated"]["pit_conditions"]
    else
        error("Please check that naming_mode is set to either 'manual' or 'generated'. \nIn the first case, write the naming convention used for each pit. \nIn the second case, write the names for the columns and the rows, such that the complete names are generated automatically.")
    end
    
    data.ignore_pits = d["pit_name_convention"]["generated"]["ignore_pits"]
    if d["which_files"]["until_file"] == -1
        data.pits_used = setdiff(data.pit_names,data.ignore_pits)[d["which_files"]["from_file"]:end]
    else
        data.pits_used = setdiff(data.pit_names,data.ignore_pits)[d["which_files"]["from_file"]:d["which_files"]["until_file"]]
    end

    data.rois = d["pit_name_convention"]["rois"]

    parts = Dict{String, Union{String,Vector{String}}}()

    parts["common_name"] = [data.common_name]          # [string]
    parts["pit_name"]    = data.pits_used               # vector of strings
    parts["roi"]         = data.rois                      # vector of strings OR string to loop over chars 
    parts["name_time_batch"]  = data.name_time_batches     # vector OR [string]
    parts["time_frame"] = [data.first_timeframe]       # [string] ****** revisar por si cambia la convención temporal

    println("Order for the naming of the files: ",parts[follow_convention[1]]," ",parts[follow_convention[2]]," ",parts[follow_convention[3]]," ",parts[follow_convention[4]]," ",parts[follow_convention[5]])

    # Names of all the files corresponding to the first time frame
    data.all_files = [i1*i2*i3*i4*i5 for i1 in parts[follow_convention[1]] for i2 in parts[follow_convention[2]] for i3 in parts[follow_convention[3]] for i4 in parts[follow_convention[4]] for i5 in parts[follow_convention[5]]]
    data.keep_rois = d["which_files"]["keep_rois"]

    data.exclude = d["which_files"]["exclude"]
    
    if length(data.all_files)>4
        println("First file to analize: ",data.all_files[1], "\nSecond file to analize: ",data.all_files[2], "\nSecond-to-last file to analize: ",data.all_files[end-1], "\nLast file to analize: ",data.all_files[end])
    else
        println("Files to analize: ",data.all_files)
    end
    
    data.report = d["save_data"]["report"]
    data.save_images = d["save_data"]["save_images"]
    return data
end
