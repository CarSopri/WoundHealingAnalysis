using CSV
using Plots
using DataFrames
using Measures

code_folder = @__DIR__

include(joinpath(code_folder,"timelapse_analyzers.jl"))

function get_excluded_rois(ignore_pits, failed_segmentations, tf)
    return unique(vcat(ignore_pits, [d for d in failed_segmentations if count(i->(i== d), failed_segmentations)>(tf*0.75)]))
end
#########################################

function handle_excluded_rois(seg_data, exclude, tf)
    
    #***esto es parte necesaria para que anden bien las cosas, tendría que pasarlo a una función
    for t in 1:tf
        for k in exclude
            delete!(seg_data.widths_distribution, k*"_T"*lpad(t,3,"0"))
            delete!(seg_data.left_walls, k*"_T"*lpad(t,3,"0"))
            delete!(seg_data.right_walls, k*"_T"*lpad(t,3,"0"))
            delete!(seg_data.left_roughnesses, k*"_T"*lpad(t,3,"0"))
            delete!(seg_data.right_roughnesses, k*"_T"*lpad(t,3,"0"))
            for (l,L_to_plot) in enumerate(seg_data.which_Ls)
                delete!(seg_data.left_roughnesses_L[t], k*"_L"*lpad(L_to_plot,4,"0"))
                delete!(seg_data.right_roughnesses_L[t], k*"_L"*lpad(L_to_plot,4,"0"))
            end
        end
    end

end


function ignore_rois_from_this_int(cutoff::Int, pit::String, n_rois::Int) #**ojo que este anda solamente para la segunda convención we
    return [pit*"ROI"*string(i) for i in cutoff+1:n_rois]
end
function ignore_rois_from_this_int(list_cutoffs,list_pits,n_rois::Int)
    return reduce(vcat, [ignore_rois_from_this_int(cutoff,pit,n_rois) for (cutoff,pit) in zip(list_cutoffs,list_pits)])
end

function load_segmentation_data(main_folder)

    # Read data
    df = CSV.read(main_folder*"scratch_report.txt", DataFrame)
    n_rows = nrow(df)
    tf = ncol(df)-1

    return df,n_rows,tf
end

function load_time_data(main_folder)
    times_df = CSV.read(main_folder*"time_report.txt", DataFrame, delim='\t')

    return [times_df[1,2:end][t] for t in 1:length(times_df[1,2:end])]
end

function find_index_t(time_batches, time)
    for i in 1:length(time_batches)
        if time ∈ time_batches[i]
            return i
        end
    end
end

Base.@kwdef mutable struct DataFromSegmentation

    pits::Vector{String} = [""]
    failed_segmentations::Vector{String} = [""]
    ignore_pits::Vector{String} = [""]

    widths_distribution = Dict{String, Vector{Float64}}()
    left_walls = Dict{String, Vector{Float64}}()
    right_walls = Dict{String, Vector{Float64}}()
    left_roughnesses = Dict{String, Vector{Float64}}()
    right_roughnesses = Dict{String, Vector{Float64}}()    
    left_roughnesses_L::Union{Nothing, Vector{Dict{String, Vector{Float64}}}} = nothing
    right_roughnesses_L::Union{Nothing, Vector{Dict{String, Vector{Float64}}}} = nothing
    which_Ls::Vector{Int64} = []

    ###
    # Colors to use to highlight the segmentation done ***capaz debería tener nombre de fondo+izq+der no más ?
    ###
    black = RGB(0.0,0.0,0.0)
    red = RGB(1.0,0.0,0.0)
    green = RGB(0.0,1.0,0.0)

    tf::Int = 1
        
end

"""
    initialize_roughness_storage(n_timepoints, which_Ls)

Initialize storage structures for L-roughness data.
"""
function initialize_roughness_storage(n_timepoints, which_Ls)
    if isempty(which_Ls)
        return nothing, nothing
    end
    left = [Dict{String, Vector{Float64}}() for _ in 1:n_timepoints]
    right = [Dict{String, Vector{Float64}}() for _ in 1:n_timepoints]
    #println("Analyzing L values: ", which_Ls)
    return left, right
end



"""
    analyze_segmentation(data_info)

Main entry point for analyzing wound healing segmentation data.

# Arguments
- `data_info`: A struct containing:
    - `main_folder::String`: Root directory for data
    - `ignore_pits::Vector{String}`: Pits to exclude from analysis
    - `pit_names::Vector{String}`: Names of all pits
    - `n_rois::Vector{Int}`: Number of ROIs per pit
    - `time_batches::Vector{Int}`: Time points to analyze
    - `name_time_batches::Vector{String}`: Names corresponding to time batches

# Returns
- `seg_data::DataFromSegmentation`: Structured analysis results
"""
function analyze_segmentation(data_info)
    
    df, n_rows, tf = load_segmentation_data(data_info.main_folder)
    times_data = load_time_data(data_info.main_folder)
    
    pit_index = 1 #**qué son estas dos variables ?
    n_photo = 1
    tf = length(times_data)
    
    seg_data = DataFromSegmentation()
    analysis_params = ParametersForAnalysis()

    # Bad data
    exclude = vcat(data_info.exclude,data_info.ignore_pits) ###***
    name_combinations = [p*roi for p in data_info.pit_names for roi in data_info.rois] #[l*string(n) for l in "ABCD" for n in 1:6] #**creo que esto cambia para el primer exp
    
    seg_data.pits = unique([p[1:2] for p in name_combinations])
    #seg_data.pits = setdiff(name_combinations, data_info.ignore_pits) #**mmm chequear esto idk
    #seg_data.pits = [p[1:2] for p in seg_data.pits]
    #seg_data.pits = unique(seg_data.pits)
    
    seg_data.ignore_pits = data_info.ignore_pits # ["B1","B2","C1","C2","D5","D6"]
    seg_data.failed_segmentations = []
    analysis_params.number_of_ROIs = length(data_info.rois) #**revisar esto
    seg_data.tf = tf

    seg_data.which_Ls = [floor(Int64,10^(1+l*0.1)) for l in 0:analysis_params.n_of_Ls]

    
    seg_data.left_roughnesses_L, seg_data.right_roughnesses_L = initialize_roughness_storage(tf, seg_data.which_Ls)
    
    
    #println(length(seg_data.pits))

#******************creo que acá tengo que arreglar el problema de que si en t!=1 no tengo segmentación pero el resto de t>1 sí, los tengo que tirar, y si tengo t=1 pero no el resto, tengo que mandar -1,-1 !!

    @info ("######################################## Excluded     ",exclude)
    @info(seg_data.pits)

    # Main analysis loops
    for pit_index in 1:length(seg_data.pits)
        for roi in 1:analysis_params.number_of_ROIs # ** creo que sería preferible loopear sobre los pits y rois directamente no? el tema está en el generate_general_ID mmmmm
            analyze_single_roi(
                pit_index, roi, analysis_params.number_of_ROIs, df, data_info, seg_data, analysis_params,
                tf, exclude, seg_data.which_Ls, 
                seg_data.left_roughnesses_L, seg_data.right_roughnesses_L
            )
        end
    end

    return seg_data
end


"""
    analyze_single_roi(pit_index, roi, n_rois, data_info, seg_data, params, tf, exclude_list, which_Ls, left_roughnesses_L, right_roughnesses_L)

Analyze a single ROI across all time points.
"""
function analyze_single_roi(pit_index, roi, n_rois, df, data_info, seg_data, params, tf, exclude_list, which_Ls, left_roughnesses_L, right_roughnesses_L)
    general_ID = generate_general_ID(pit_index, roi, n_rois, df)  # Need df from load_segmentation_data
    
    if general_ID in exclude_list
        @info ("Ignoring: ", general_ID)
        return
    end
    
    @info ("Analyzing: ", general_ID)
    
    # Initialize histograms if needed
    hist_horizontal, hist_vertical = params.plot_hist ? init_histograms() : (nothing, nothing)
    
    # Check if first time point exists
    t1_exists = check_time_point_exists(general_ID, 1, data_info)

    #if !t1_exists
    #    push!(seg_data.ignore_pits, general_ID)  # Add to seg_data 
    #    @info ("File ignored: t1 does not exist for ", general_ID) #*************bueno esto lo tengo que corregir
    #    return
    #end
    
    # Analyze each time point
    for time in 1:tf
        analyze_time_point(
            time, general_ID, data_info, seg_data, params,
            which_Ls, left_roughnesses_L, right_roughnesses_L,
            hist_horizontal, hist_vertical, tf
        )
    end
    
    # Finalize plots if needed
    if params.plot_hist
        finalize_and_display_plots!(hist_horizontal, hist_vertical, general_ID, data_info.main_folder)
    end
end

"""
    generate_general_ID(pit, roi, data_info, df)

Generate the general ID for a pit-ROI combination.
"""
function generate_general_ID(pit_index, roi, n_rois, df)
    # This needs to be adjusted based on your actual data structure
    row_index = (pit_index - 1) * n_rois + roi
    return df[row_index, 1]
end

"""
    check_time_point_exists(general_ID, time, data_info)

Check if segmentation image exists for a specific time point.
"""
function check_time_point_exists(general_ID, time, data_info)
    image_path = generate_image_path(general_ID, time, data_info)
    println("File found: ",image_path)
    return isfile(image_path)
end

"""
    generate_image_path(general_ID, time, data_info)

Generate the file path for a wound edges image.
"""
function generate_image_path(general_ID, time, data_info) #******chequear esto
    ID = generate_image_ID(general_ID, time, data_info)
    return data_info.main_folder * "wound_edges_image_" * ID * ".png"
end

function generate_image_ID(general_ID, time, data_info)
    time_batch_name = data_info.name_time_batches[find_index_t(data_info.time_batches, time)]
    return general_ID[1:2] * "ROI" * general_ID[end] * time_batch_name  * "_T" * lpad(time, 3, "0")
end

function handle_failed_segmentation(general_ID, time, which_Ls, left_roughnesses_L, right_roughnesses_L, failed_segmentations)
    push!(failed_segmentations,general_ID)
    
    for (l,L_to_plot) in enumerate(which_Ls)
        left_roughnesses_L[time][general_ID*"_L"*lpad(L_to_plot,4,"0")] = [-1.0, -1.0]
        right_roughnesses_L[time][general_ID*"_L"*lpad(L_to_plot,4,"0")] = [-1.0, -1.0]
    end
    
end

"""
    analyze_time_point(time, general_ID, t1_exists, data_info, seg_data, params, which_Ls, left_roughnesses_L, right_roughnesses_L, hist_horizontal, hist_vertical, tf)

Analyze a single time point for a given ROI.
"""
function analyze_time_point(time, general_ID, data_info, seg_data, params, which_Ls, left_roughnesses_L, right_roughnesses_L, hist_horizontal, hist_vertical, tf)
    
    
    ID = generate_image_ID(general_ID, time, data_info)
    ID_for_data = general_ID*"_T"*lpad(time,3,"0")
    image_path = generate_image_path(general_ID, time, data_info)
    """
    # Corrección a datos fibroblastos-emilio-may2026
    if occursin("A4",image_path)
        image_path = replace(image_path, "A4" => "B1", count=1)           # **************************************************
    elseif occursin("A5",image_path)
        image_path = replace(image_path, "A5" => "B2", count=1)
    elseif occursin("A6",image_path)
        image_path = replace(image_path, "A6" => "B3", count=1)
    elseif occursin("B1",image_path)
        image_path = replace(image_path, "B1" => "A4", count=1)
    elseif occursin("B2",image_path)
        image_path = replace(image_path, "B2" => "A5", count=1)
    elseif occursin("B3",image_path)
        image_path = replace(image_path, "B3" => "A6", count=1)
    elseif occursin("C4",image_path)
        image_path = replace(image_path, "C4" => "D1", count=1)
    elseif occursin("C5",image_path)
        image_path = replace(image_path, "C5" => "D2", count=1)
    elseif occursin("C6",image_path)
        image_path = replace(image_path, "C6" => "D3", count=1)
    elseif occursin("D1",image_path)
        image_path = replace(image_path, "D1" => "C4", count=1)
    elseif occursin("D2",image_path)
        image_path = replace(image_path, "D2" => "C5", count=1)
    elseif occursin("D3",image_path)
        image_path = replace(image_path, "D3" => "C6", count=1)
    end
    """
    """
    # Corrección a datos queratinocitos-italianos-jun2026
    if occursin("A7",image_path)
        image_path = replace(image_path, "A7" => "C5", count=1)           # **************************************************
    elseif occursin("B7",image_path)
        image_path = replace(image_path, "B7" => "D5", count=1)
    elseif occursin("A8",image_path)
        image_path = replace(image_path, "A8" => "C6", count=1)
    elseif occursin("B8",image_path)
        image_path = replace(image_path, "B8" => "D6", count=1)
    end
    """
    # Corrección a datos queratinocitos-jun2026

    
    if !isfile(image_path)
        @info ("Segmentation failed for ", ID)
        handle_failed_segmentation(general_ID, time, which_Ls, left_roughnesses_L, right_roughnesses_L, seg_data.failed_segmentations)
        return
    end
    
    process_successful_segmentation(
        ID_for_data, general_ID, time, image_path, data_info, params, seg_data,
        which_Ls, left_roughnesses_L, right_roughnesses_L,
        hist_horizontal, hist_vertical, tf
    )
end

"""
    process_successful_segmentation(...)

Process a successfully segmented image.
"""
function process_successful_segmentation(ID, general_ID, time, image_path, data_info, params, seg_data, which_Ls, left_roughnesses_L, right_roughnesses_L, hist_horizontal, hist_vertical, tf)
    im = load(image_path)
    _, w = size(im) #***********sospecho fuertemente que no necesito esto
    width_threshold = w * 3 / 5 #***ver de dónde salía esto !!! poner en params para análisis probablemente
    color_wound = params.black #**en vez de llamarle black, llamarle color_wound directamente en ParametersForAnalysis !
    
    widths, left_wall_ID, right_wall_ID = get_row_widths_and_walls(im, color_wound)
    
    mean_width = mean(widths)
    std_width = std(widths)
    valid_wound = is_valid_wound(widths, mean_width, width_threshold)

    if (nothing in left_wall_ID) || (nothing in right_wall_ID)
        valid_wound = false
    end
    
    if !valid_wound
        @info ("!!!!!!!!!!!!!!!! For some reason wound is not valid  ", mean(widths), "  ",ID,"  ",width_threshold)
        push!(seg_data.failed_segmentations, general_ID)
    end
    
    # Process based on analysis type
    if !params.plot_L
        process_standard_analysis(ID, left_wall_ID, right_wall_ID, #w, 
            mean_width, std_width, valid_wound, params, seg_data)
    else
        process_multi_scale_analysis(ID, general_ID, time, left_wall_ID, right_wall_ID, #w, 
            mean_width, std_width, valid_wound, which_Ls, left_roughnesses_L, right_roughnesses_L, params, seg_data)
    end
    
    # Plotting if needed
    if mean_width < width_threshold && params.plot_hist
        add_histogram_traces!(
                            hist_horizontal, hist_vertical,
                            time, tf, ID_for_data,
                            widths,
                            map(x -> x::Int64, filter(!isnothing, left_wall_ID)), map(x -> x::Int64, filter(!isnothing, right_wall_ID)),
                            mean_width, std_width,
                            left_wall[1], left_wall[2],
                            right_wall[1], right_wall[2]
                        )
    end
end

"""
    is_valid_wound(widths, mean_width, width_threshold)

Determine if the wound segmentation is valid.
"""
function is_valid_wound(widths, mean_width, width_threshold)
    return !any(x -> x > width_threshold, widths)&& !any(x -> x < 40, widths) #**ese 40 debería ser un parámetro para el análisis (estoy pidiendo que si está casi cerrado, lo ingnore)
end # (mean_width < width_threshold)

"""
    process_standard_analysis(...)

Process standard analysis without L-roughness.
"""
function process_standard_analysis(ID, left_wall_ID, right_wall_ID, #w, 
        mean_width, std_width, valid_wound, params, seg_data)
    left_wall = summarize_wall_data(left_wall_ID, #w, 
                                params.L_for_roughness)
    right_wall = summarize_wall_data(right_wall_ID, #w, 
                                params.L_for_roughness)
    
    store_results!(
        (seg_data.widths_distribution, seg_data.left_walls, seg_data.right_walls, seg_data.left_roughnesses, seg_data.right_roughnesses),
        ID,
        (mean_width, std_width),
        left_wall[1:2],
        right_wall[1:2],
        (left_wall[3], left_wall[4]),
        (right_wall[3], right_wall[4]),
        valid_wound
    )
end

"""
    process_multi_scale_analysis(...)

Process analysis with L-roughness at multiple scales.
"""
function process_multi_scale_analysis(ID, general_ID, time, left_wall_ID, right_wall_ID, #w, 
        mean_width, std_width, valid_wound, which_Ls, left_roughnesses_L, right_roughnesses_L, params, seg_data)
    # Store base analysis
    process_standard_analysis(ID, left_wall_ID, right_wall_ID, #w, 
                        mean_width, std_width, valid_wound, params, seg_data)
    
    # Store multi-scale roughness
    for (l, L_to_plot) in enumerate(which_Ls)
        left_wall = summarize_wall_data(left_wall_ID, #w, 
                                        L_to_plot)
        right_wall = summarize_wall_data(right_wall_ID, #w, 
                                        L_to_plot)
        
        roughness_key = general_ID * "_L" * lpad(L_to_plot, 4, "0")
        roughness_value = valid_wound ? [left_wall[3], left_wall[4]] : [-1.0, -1.0]
        
        left_roughnesses_L[time][roughness_key] = valid_wound ? [left_wall[3], left_wall[4]] : [-1.0, -1.0]
        right_roughnesses_L[time][roughness_key] = valid_wound ? [right_wall[3], right_wall[4]] : [-1.0, -1.0]
    end
end
