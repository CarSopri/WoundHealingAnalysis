
using TOML

Base.@kwdef mutable struct DataInfo

    main_folder::String = @__DIR__
    subfolders::Vector{String} = [""]
            
    file_format::String = ".tif"

    conditions::Vector{String} = ["control"]

    # The number of elements in subfolders has to be the same as in time_series and name_time_series
    time_series::Vector{Vector{Int}} = []
    name_time_series::Vector{String} = []
    first_timepoint::String = "001"

    channels::Union{String,Vector{String}} = ["1_"]
    common_name::String = ""
    follow_convention::Vector{String} = ["common_name","roi","name_time_series","well_name","channel","timepoint"]

    rois::Union{String,Vector{String}} = ""

    row_names::Union{String,Vector{String}} = ""
    col_names::Union{String,Vector{String}} = ""
    well_conditions::Vector{Int} = []

    well_names::Vector{String} = []
    ignore_wells::Vector{String} = []
    wells_used::Vector{String} = []
    keep_rois::Vector{Int} = [] #***acomodar esto, el de arriba saca wells directamente y este deja rois. also no sé si esto anda si está vacío ?
    exclude::Vector{String} = []


    all_files::Vector{String} = []

    report::Bool = false
    save_images::Bool = false
end

function ignore_rois_from_this_int(cutoff::Int, well::String, n_rois::Int) #**ojo que este anda solamente para la segunda convención we
    return [well*"ROI"*string(i) for i in cutoff+1:n_rois]
end
function ignore_rois_from_this_int(list_cutoffs,list_wells,n_rois::Int)
    return reduce(vcat, [ignore_rois_from_this_int(cutoff,well,n_rois) for (cutoff,well) in zip(list_cutoffs,list_wells)])
end

const FILENAME_COMPONENTS = Set([
    "common_name",
    "roi",
    "name_time_series",
    "well_name",
    "channel",
    "timepoint",
])

function validate_naming_convention(data::DataInfo)
    unknown = setdiff(data.follow_convention, FILENAME_COMPONENTS)

    isempty(unknown) ||
        error("Unknown filename component(s) in follow_convention: $unknown")

    length(data.follow_convention) == length(FILENAME_COMPONENTS) ||
        error("follow_convention must contain each filename component exactly once.")

    length(unique(data.follow_convention)) == length(data.follow_convention) ||
        error("follow_convention contains duplicate components.")

    length(data.time_series) == length(data.name_time_series) ||
        error("time_series and name_time_series must have the same length.")

    length(data.time_series) == length(data.subfolders) ||
        error("time_series and subfolders must have the same length.")

    return nothing
end

function normalize_user_path(path::AbstractString)
    return replace(strip(path), "\\" => "/")
end

function resolve_path(path, config_path)
    config_dir = dirname(abspath(config_path))

    if isabspath(path)
        return normpath(path)
    else
        return normpath(joinpath(config_dir, path))
    end
end

function validate_folders(main_folder, subfolders)
    #if !isabspath(main_folder)
    #    error("main_folder must be an absolute path.\nReceived: $(main_folder)")
    #end


    if !isdir(main_folder)
        error("The main data folder does not exist:\n$(main_folder)\nPlease check the [folders] section in config.toml. Use forward slashes / to separate directories.")
    end

    for subfolder in subfolders
        if isabspath(subfolder)
            error(
                "Each subfolder must be relative to main_folder.\n" *
                "Received: $subfolder"
            )
        end

        folder = joinpath(joinpath(main_folder, subfolder))

        #if !isdir(folder)
        #    error(
        #        "Could not find a configured data folder:\n" *
        #        "$folder\n" *
        #        "Please check the subfolders list in config.toml."
        #    )
        #end
    end

    return nothing
end

#data.keep_rois = ignore_rois_from_this_int(d["file_selection"]["keep_rois"],
#                                            [l for l in data.well_names],length(data.rois)
#                                            )

function load_data_from_config_file(config_file)

    d = TOML.parsefile(config_file)
    data = DataInfo()

    data.main_folder = resolve_path(d["folders"]["main_folder"], config_file)
    data.subfolders = d["folders"]["subfolders"]

    data.main_folder = normalize_user_path(data.main_folder)
    data.subfolders = [normalize_user_path(subfolder) for subfolder in data.subfolders]

    validate_folders(data.main_folder, data.subfolders)
            
    data.file_format = d["format"]["file_format"]
    data.conditions = d["conditions"]["conditions"]

    # The number of elements in subfolders has to be the same as in time_series and name_time_series
    data.time_series = d["timepoint_naming_convention"]["time_series"]
    data.name_time_series = d["timepoint_naming_convention"]["name_time_series"]
    data.first_timepoint = d["timepoint_naming_convention"]["first_timepoint"]

    data.channels = d["general_naming_convention"]["channels"]
    data.common_name = d["general_naming_convention"]["common_name"]
    data.follow_convention = split(d["general_naming_convention"]["follow_convention"],"*")
    follow_convention = data.follow_convention

    validate_naming_convention(data)

    data.rois = d["well_naming_convention"]["rois"]

    if d["well_naming_convention"]["well_naming_mode"] == "manual"
        data.row_names = d["well_naming_convention"]["manual"]["row"]
        data.col_names = d["well_naming_convention"]["manual"]["col"]
        data.well_names = d["well_naming_convention"]["manual"]["well_names"]
        data.well_conditions = d["well_naming_convention"]["manual"]["well_conditions"]
        data.ignore_wells = []
    elseif d["well_naming_convention"]["well_naming_mode"] == "generated"
        data.row_names = d["well_naming_convention"]["generated"]["row"]
        data.col_names = d["well_naming_convention"]["generated"]["col"]
        data.well_names = [l*n for l in data.row_names for n in data.col_names]
        data.well_conditions = d["well_naming_convention"]["generated"]["well_conditions"]
        data.ignore_wells = d["well_naming_convention"]["generated"]["ignore_wells"]
    else
        error("Please check that well_naming_mode is set to either 'manual' or 'generated'. \nIn the first case, write the naming convention used for each well. \nIn the second case, write the names for the columns and the rows, such that the complete names are generated automatically.")
    end
    

    if d["file_selection"]["until_file"] == -1
        data.wells_used = setdiff(data.well_names,data.ignore_wells)[d["file_selection"]["from_file"]:end]
    else
        data.wells_used = setdiff(data.well_names,data.ignore_wells)[d["file_selection"]["from_file"]:d["file_selection"]["until_file"]]
    end


    parts = Dict{String, Union{String,Vector{String}}}()

    parts["common_name"] = [data.common_name]          # [string]
    parts["well_name"]    = data.wells_used               # vector of strings
    parts["roi"]         = data.rois                      # vector of strings OR string to loop over chars 
    parts["name_time_series"]  = [data.name_time_series[1]]     # vector OR [string]
    parts["timepoint"] = [data.first_timepoint]       # [string] ****** revisar por si cambia la convención temporal
    parts["channel"]    = data.channels

    println("Order for naming the files: ",parts[follow_convention[1]]," ",parts[follow_convention[2]]," ",parts[follow_convention[3]]," ",parts[follow_convention[4]]," ",parts[follow_convention[5]]," ",parts[follow_convention[6]])

    # Names of all the files corresponding to the first time point
    data.all_files = [i1*i2*i3*i4*i5*i6 for i1 in parts[follow_convention[1]] for i2 in parts[follow_convention[2]] for i3 in parts[follow_convention[3]] for i4 in parts[follow_convention[4]] for i5 in parts[follow_convention[5]] for i6 in parts[follow_convention[6]]]
    data.keep_rois = d["file_selection"]["keep_rois"]

    data.exclude = d["file_selection"]["exclude"]
    
    if length(data.all_files)>4
        println("First file to analize: ",data.all_files[1], "\nSecond file to analize: ",data.all_files[2], "\nSecond-to-last file to analize: ",data.all_files[end-1], "\nLast file to analize: ",data.all_files[end])
    else
        println("Files to analize (first frame name only): ",data.all_files)
    end
    
    data.report = d["save_data"]["report"]
    data.save_images = d["save_data"]["save_images"]

    """
    Configuration summary
---------------------
Image directory: /data/my_experiment
Image format: .tif
Naming mode: generated
Wells selected: 18
Conditions: control, treatment
ROIs excluded: 2
Save report: yes
Save segmented images: no

Configuration valid. Ready to begin.

    
     """



    return data
end
