code_folder = @__DIR__
include(joinpath(code_folder,"load_data.jl"))
include(joinpath(code_folder,"save_data.jl"))
include(joinpath(code_folder,"segment_image.jl"))

function initialize_report_files(data, report)
    if !report
        return 0, 0, 0
    end
    
    area_file = init_report_file(data.main_folder * "scratch_report.txt", data.time_series)
    width_file = init_report_file(data.main_folder * "width_report.txt", data.time_series)
    time_file = init_report_file(data.main_folder * "time_report.txt", data.time_series)
    
    return area_file, width_file, time_file
end

function save_report_files(data, report_file, time_file)
    CSV.write(data.main_folder * "scratch_report.txt", report_file, delim='\t')
    CSV.write(data.main_folder * "time_report.txt", time_file, delim='\t')
end

function update_report_file_ids(report_file, time_file, well, roi, total_time_frames)
    well_roi_id = well * string(roi)
    
    # Add to scratch report if not present
    if well_roi_id ∉ report_file.ID
        push!(report_file, vcat([well_roi_id], [0.0 for _ in 1:total_time_frames]))
    end
    
    # Add to time report if not present
    if well_roi_id ∉ time_file.ID
        push!(time_file, vcat([well_roi_id], [0.0 for _ in 1:total_time_frames]))
    end
end

function update_time_report(data, time_file, well, roi, imfile, full_id)
    # Reference file is the first time point of the first series #**acá también, ojo con el orden del namefile porque debería venir de config !!! T.T
    ref = data.main_folder * data.subfolders[1] *
          well * string(roi) * data.name_time_series[1] *
          data.common_name * lpad(1, 3, "0") * data.file_format
    
    time_from_file = (mtime(imfile) - mtime(ref)) / 3600  # in hours
    
    # Find the correct row and column to update
    row = findfirst(time_file.ID .== full_id[1:end-12])
    col = parse(Int, full_id[end-2:end])
    
    time_file[row, col + 1] = time_from_file
end


function first_pass_collect_cuts(data, well, roi, report_file, time_file, tf, save_images, segmentation_hyperparams, report)
    cuts_per_time = Dict{Int, Tuple{Int,Int}}()
    initial_cut = (nothing, nothing)  # Start with no cut

    # Update report files with new IDs if needed
    if report
        update_report_file_ids(report_file, time_file, well, roi, tf)
    end
    
    for (series_idx, time_series) in enumerate(data.time_series)
        general_id = well * string(roi) * data.name_time_series[series_idx] #**ojo acá y en ref porque estoy cambiando el orden respecto del primer exp we
        
        for time in time_series
            # Build file paths and IDs
            subfolder = data.subfolders[series_idx]
            frame_number = time - time_series[1] + 1
            general_name = general_id * data.common_name * lpad(frame_number, 3, "0") * data.file_format
            imfile = joinpath(data.main_folder, subfolder, general_name)
            full_id = general_id * "_T" * lpad(time, 3, "0")
            
            @info "############################## Processing: $full_id ##############################"

            if !isfile(imfile)
                @info (imfile, "\n!!!!!!!!!!!!!!!!!!!!!! FILE NOT FOUND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break
            end
                
            # Update time report if needed
            if report
                update_time_report(data, time_file, well, roi, imfile, full_id)
            end

            k = 0
            
            # Run segmentation for this frame
            y_cut, keep_running = run_segmentation(
                data.main_folder, full_id, 
                imfile, 
                report_file, save_images, initial_cut, segmentation_hyperparams
            )

            # In case the segmentation wasn't right
            while keep_running && k<15 #***debería ser un input parameter
                y_cut, keep_running = run_segmentation(data.main_folder, full_id, data.main_folder*subfolder*general_name, report_file, save_images, y_cut, segmentation_hyperparams)
                k+=1
                @info ("-----> Number of iterations: ",k, ", Cut in y (if shadow present): ",y_cut)
            end
            
            if y_cut[1] == y_cut[2] # Whole image was detected as shadow
                @info ("!!!!!!!!!!!!!!!! No segmentation was possible. Seems like the whole image is in shadow. !!!!!!!!!!!!!!!!")
                if time==1
                    break
                end
            end
            
            # Update initial cut if this is the first time point 
            if time == 1 && y_cut != [1, 1960] #**1960==h_original en realidad, corregir
                initial_cut = y_cut
                @info "Initial cut set to: $initial_cut"
            end
            # Store the cut for this time point
            cuts_per_time[time] = y_cut
            
            
        end
    end
    
    return cuts_per_time
end





function analyze_if_refinement_needed(cuts_per_time)
    # Filter out failed segmentations
    valid_cuts = [(time, cut) for (time, cut) in cuts_per_time if cut != [-1, -1]]
    
    isempty(valid_cuts) && return false, nothing
    
    # Extract all cut bounds
    lower_bounds = [cut[1] for (_, cut) in valid_cuts]
    upper_bounds = [cut[2] for (_, cut) in valid_cuts]
    
    # Find the most aggressive cut needed
    max_lower = maximum(lower_bounds)   # Largest lower bound (cut more from bottom)
    min_upper = minimum(upper_bounds)   # Smallest upper bound (cut more from top)
    
    # Get the cut used initially (from first valid frame)
    first_cut = cuts_per_time[1]
    current_lower = first_cut[1] != -1 ? first_cut[1] : 1 #No contamos las segmentaciones fallidas
    current_upper = first_cut[2] != -1 ? first_cut[2] : 1960
    
    # Count frames that need a larger cut than the initial one
    frames_needing_larger_cut = 0
    for (_, cut) in valid_cuts
        if cut[1] > current_lower || cut[2] < current_upper
            frames_needing_larger_cut += 1
        end
    end
    
    # Threshold: more than 1/3 of frames need larger cut
    threshold = length(valid_cuts) / 3
    needs_refinement = frames_needing_larger_cut > threshold
    
    if needs_refinement
        @info "Analysis: $frames_needing_larger_cut out of $(length(valid_cuts)) frames need larger cut"
        @info "Current cut: ($current_lower, $current_upper)"
        @info "Proposed cut: ($max_lower, $min_upper)"
        return true, (max_lower, min_upper)
    end
    
    return false, nothing
end

function second_pass_refine_cuts(data, well, roi, better_cut, old_cuts,
                                  report_file, time_file, save_images, 
                                  segmentation_hyperparams, report)
    current_lower, current_upper = better_cut
    
    for (series_idx, time_series) in enumerate(data.time_series)
        general_id = well * string(roi) * data.name_time_series[series_idx]
        
        for time in time_series
            # Only re-segment frames that actually need the larger cut
            old_cut = old_cuts[time]
            needs_larger_cut = old_cut[1] > current_lower || old_cut[2] < current_upper
            
            if needs_larger_cut
                # Build file paths
                frame_number = time - time_series[1] + 1
                general_name = general_id * data.common_name * lpad(frame_number, 3, "0") * data.file_format
                imfile = data.main_folder * data.subfolders[series_idx] * general_name
                full_id = general_id * "_T" * lpad(time, 3, "0")
                
                @info "Re-segmenting $full_id with better cut: $better_cut"

                #**capaz lo que sigue (la segmentación hasta k veces) debería ser una función también)
                k = 0

                
                # Run segmentation again with the better cut
                y_cut_new, keep_running = run_segmentation(
                    data.main_folder, full_id, 
                    imfile,
                    report_file, save_images, better_cut, segmentation_hyperparams
                )

                # In case the segmentation wasn't right
                while keep_running && k<15 #***debería ser un input parameter
                    y_cut, keep_running = run_segmentation(data.main_folder, full_id, data.main_folder*subfolder*general_name, report_file, save_images, y_cut, segmentation_hyperparams)
                    k+=1
                    @info ("-----> Number of iterations: ",k, ", Cut in y (if shadow present): ",y_cut)
                end

                if y_cut[1] == y_cut[2] # Whole image was detected as shadow
                    @info ("!!!!!!!!!!!!!!!! No segmentation was possible. Seems like the whole image is in shadow. !!!!!!!!!!!!!!!!")
                    if time == 1
                        break
                    end
                end
                
                # Log if the cut changed further
                if y_cut_new != old_cut
                    @info "Cut evolved from $old_cut to $y_cut_new" #****pensar qué onda si pasa esto, sobre todo para el primer t
                end
                
            end
        end
    end
end

function process_well_roi_combination(data, well, roi, report_file, time_file, tf, save_images, segmentation_hyperparams, report)
    # Store cuts found for each time point
    cuts_per_time = Dict{Int, Vector{Int}}()
    
    # First pass: collect all cuts without re-segmentation
    cuts_per_time = first_pass_collect_cuts(
        data, well, roi, report_file, time_file, tf,
        save_images, segmentation_hyperparams, report
    )
    
    # Analyze if refinement is needed
    refinement_needed, better_cut = analyze_if_refinement_needed(cuts_per_time)
    
    # Second pass: re-segment if needed
    if refinement_needed
        @info "Refinement needed! Re-segmenting with better cut: $better_cut"
        second_pass_refine_cuts(
            data, well, roi, better_cut, cuts_per_time,
            report_file, time_file, save_images, segmentation_hyperparams, report
        )
    end
    
    # Save after each well/roi combination
    if report
        save_report_files(data, report_file, time_file)
    end
end
    

function run_segmentation_for_all_files_new(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters)
    report = data.report
    save_images = data.save_images

    # Get list of wells to process (excluding ignored ones)
    wells_to_process = [p for p in data.well_names if p ∉ data.ignore_wells]
    tf = data.time_series[end][end]

    ID = String
    general_name = String
    subfolder = String
    
    # Si no existe el file, crearlo, y si existe, hacer un df
    area_file, width_file, time_file = initialize_report_files(data, report)

    l = 0
    for well in wells_to_process
        for roi in data.rois
            l+=1
            if l!=1 #****************** esto debería tener en cuenta el file_start o algo así que puse en config !!
                continue
            end

            process_well_roi_combination(
                data, well, roi, area_file, time_file, tf,
                save_images, segmentation_hyperparams, report
            )
            
        end
    end
    
    if report
        save_report_files(data, area_file, time_file)
    end
end

function well_roi_id(well::String, roi::String)
    roi_label = startswith(roi, "ROI") ? roi : "ROI" * roi
    return well * roi_label
end
function image_filename(data::DataInfo, well::String, roi::String, channel::String, series_index::Int, time::Int)
    series = data.time_series[series_index]

    # Preserve the original convention: filenames use a frame index
    # relative to the beginning of each time series.
    frame_number = time - first(series) + 1
    timepoint = lpad(string(frame_number), 3, '0')

    parts = Dict(
        "common_name"      => data.common_name,
        "roi"              => roi,
        "name_time_series" => data.name_time_series[series_index],
        "well_name"        => well,
        "channel"          => channel,
        "timepoint"        => timepoint,
    )

    return join((parts[key] for key in data.follow_convention), "") *
           data.file_format
end
function image_path(data::DataInfo, well::String, roi::String, channel::String, series_index::Int, time::Int)
    filename = image_filename(data, well, roi, channel, series_index, time)

    return joinpath(
        data.main_folder,
        data.subfolders[series_index],
        filename,
    )
end
function segment_one_image(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters, image_id::String, filepath::String, area_file, y_cut_all_t, time::Int)
    save_images = data.save_images
    k = 0

    # First segmentation attempt.
    y_cut, keep_running = run_segmentation(
        data.main_folder,
        image_id,
        filepath,
        area_file,
        save_images,
        y_cut_all_t,
        segmentation_hyperparams,
    )

    if time == 1
        y_cut_all_t = y_cut
    end

    # Second attempt when the first result indicates a cut was found.
    if y_cut != (1, 1960)
        y_cut, keep_running = run_segmentation(
            data.main_folder,
            image_id,
            filepath,
            area_file,
            save_images,
            y_cut_all_t,
            segmentation_hyperparams,
        )

        if y_cut == (-1, -1)
            return y_cut_all_t, true
        end

        if time == 1
            y_cut_all_t = y_cut
        end
    end

    # Retry if the segmentation has not yet been accepted.
    while keep_running && k < 15
        y_cut, keep_running = run_segmentation(
            data.main_folder,
            image_id,
            filepath,
            area_file,
            save_images,
            y_cut_all_t,
            segmentation_hyperparams,
        )

        k += 1

        @info "Segmentation retry" image_id iteration=k y_cut=y_cut

        if time == 1
            y_cut_all_t = y_cut
        end
    end

    # The Boolean indicates whether to stop the current time series.
    return y_cut_all_t, false
end

function ensure_report_row!(report_file, report_id::String, n_timepoints::Int)
    if !(report_id in report_file.ID)
        push!(report_file, vcat([report_id], zeros(n_timepoints)))
    end

    return nothing
end


function save_segmentation_reports(data::DataInfo, area_file, time_file)
    CSV.write(
        joinpath(data.main_folder, "scratch_report.txt"),
        area_file;
        delim = '\t',
    )

    CSV.write(
        joinpath(data.main_folder, "time_report.txt"),
        time_file;
        delim = '\t',
    )

    return nothing
end

function roi_labels(rois::Union{String,Vector{String}})
    return rois isa String ? (isempty(rois) ? String[] : [rois]) : rois
end
function segment_all_files(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters)
    validate_naming_convention(data)

    # Prepare reports and basic information.
    area_file, width_file, time_file = initialize_report_files(data, data.report)

    n_timepoints = data.time_series[end][end]
    rois = roi_labels(data.rois) #***nidea

    # Process each channel, well, ROI, and time series.
    for channel in data.channels
        for well in data.wells_used
            for roi in rois
                report_id = well_roi_id(well, roi)

                # Shared cut information for this well/ROI/channel,
                # carried across the time points and time series.
                y_cut_all_t = (nothing, nothing)

                if data.report
                    ensure_report_row!(area_file, report_id, n_timepoints)
                    ensure_report_row!(time_file, report_id, n_timepoints)
                end

                for (series_index, times) in enumerate(data.time_series)
                    #series_label = data.name_time_series[series_index]
                    general_ID = report_id

                    for time in times
                        filename = image_filename(data, well, roi, channel, series_index, time)
                        filepath = joinpath(data.main_folder, data.subfolders[series_index], filename)
                        image_id = general_ID * "_T" * lpad(string(time), 3, '0')

                        # Skip missing images, as in the original function.
                        if !isfile(filepath)
                            @info "FILE NOT FOUND" filepath
                            continue
                        end

                        @info "Processing image" image_id filepath

                        # Update the elapsed-time report.
                        if data.report
                            reference_path = image_path(data, well, roi, channel, 1, first(data.time_series[1]))
                            time_from_file = (mtime(filepath) - mtime(reference_path)) / 3600
                            row = findfirst(==(report_id), time_file.ID)

                            if isnothing(row)
                                error("Report row not found for ID: $report_id")
                            end

                            time_file[row, time + 1] = time_from_file
                        end

                        # Run the complete segmentation sequence.
                        y_cut_all_t, stop_series = segment_one_image(
                            data,
                            segmentation_hyperparams,
                            image_id,
                            filepath,
                            area_file,
                            y_cut_all_t,
                            time,
                        )

                        # Preserve the original break behavior:
                        # stop this time series, then continue with the next one.
                        if stop_series
                            break
                        end
                    end
                end

                # Save reports after finishing each well/ROI/channel combination.
                if data.report
                    save_segmentation_reports(data, area_file, time_file)
                end
            end
        end
    end

    # Final report write, as in the original function.
    if data.report
        save_segmentation_reports(data, area_file, time_file)

        # TODO: Generate the width report with standard deviations.
    end

    return nothing
end

function segment_all_files2(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters)
    report = data.report
    save_images = data.save_images
    
    well = [p for p in data.well_names if p ∉ data.ignore_wells]
    tf = data.time_series[end][end]

    ID = String
    general_name = String
    subfolder = String
    
    area_file, width_file, time_file = initialize_report_files(data, report)

    l = 0
    
    for ch in data.channels
        for p in well
            for n_photo in data.rois
                l+=1

                y_cut_all_t = (nothing,nothing)

                for (n_series,time_series) in enumerate(data.time_series)

                    general_ID = p*string(n_photo)*data.name_time_series[n_series] #**ojo acá y en ref porque estoy cambiando el orden respecto del primer exp we

                    if report

                        # Check if ID is already a row in report_file
                        #**acá debería ser p*string(n_photo); lo del medio debería cambiar sólo con t así que no hay que repetirlo en rows
                        if p*string(n_photo) ∉ area_file.ID #**acá ojooo es p para el primer exp
                            push!(area_file, vcat([p*string(n_photo)],[0.0 for i in 1:tf]))
                        end
                        if p*string(n_photo) ∉ time_file.ID
                            push!(time_file, vcat([p*string(n_photo)],[0.0 for i in 1:tf]))
                        end
                    end

                    for time in time_series

                        #time_displacement = n_series!=1 ? time_series[n_series-1][end] : 0
                        subfolder = data.subfolders[n_series]
                        general_name = general_ID*data.common_name*lpad(time-time_series[1]+1,3,"0")*data.file_format
                        ID = general_ID*"_T"*lpad(time,3,"0")

                        if !isfile(data.main_folder*subfolder*general_name)
                            @info (data.main_folder*subfolder*general_name, "\n!!!!!!!!!!!!!!!!!!!!!! FILE NOT FOUND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            continue
                        end

                        @info ("#####################   ",ID,"   #####################")

                        if report
                            ref = data.main_folder*data.subfolders[1]*p*string(n_photo)*data.name_time_series[1]*data.common_name*lpad(1,3,"0")*data.file_format
                            time_from_file = ( mtime(data.main_folder*subfolder*general_name) - mtime(ref) )/3600 # in hours

                            @info (time_file.ID,"  ",ID)
                            row = findfirst(time_file.ID .== ID[1:end-12])
                            col = parse(Int64,ID[end-2:end])
                            @info (data.main_folder*subfolder*general_name,"  ",ref)
                            #@info (row, col, "  ",time_from_file)
                            time_file[row, col+1] = time_from_file
                        end

                        # **si ya se halló un y_cut en t=1, hay que aplicarlo al resto
                        # si se halló después, mejor ignorar ese solamente                #####################################
                        k = 0
                        # **va a encontrar el lente a la primera (no espero que dé mal la primera segmentación en ese caso) xq está pintado de blanco
                        y_cut, keep_running = run_segmentation(data.main_folder, ID, data.main_folder*subfolder*general_name, area_file, save_images,y_cut_all_t, segmentation_hyperparams)

                        if time == 1
                            y_cut_all_t = y_cut
                        end
                        # Esperaría que en el segundo paso no recorte más xq ya está, pero que segmente mejor
                        # Si no había lente presente, este paso es redundante, sólo repite el anterior
                        if y_cut != (1,1960)
                            y_cut, keep_running = run_segmentation(data.main_folder, ID, data.main_folder*subfolder*general_name, area_file, save_images, y_cut_all_t, segmentation_hyperparams)
                            if y_cut == (-1,-1)
                                break
                            end
                            if time == 1
                                y_cut_all_t = y_cut
                            end
                        end
                        # In case the segmentation wasn't right
                        while keep_running && k<15 #***debería ser un input parameter
                            y_cut, keep_running = run_segmentation(data.main_folder, ID, data.main_folder*subfolder*general_name, area_file, save_images, y_cut_all_t, segmentation_hyperparams)
                            k+=1
                            @info ("-----> Number of iterations: ",k, ", Cut in y (if shadow present): ",y_cut)
                            if time == 1
                                y_cut_all_t = y_cut
                            end
                        end
                    end
                end

                if report
                    CSV.write(data.main_folder*"scratch_report.txt", area_file, delim='\t') #***esto lo tendría que adaptar para que vaya escribiendo cada tanto aunque sea idk; capaz un log que anote de dónde hasta dónde corrió x las dudas?
                    CSV.write(data.main_folder*"time_report.txt", time_file, delim='\t')
                end

            end
        end
    end
    
    if report
        CSV.write(data.main_folder*"scratch_report.txt", area_file, delim='\t') #***esto lo tendría que adaptar para que vaya escribiendo cada tanto aunque sea idk; capaz un log que anote de dónde hasta dónde corrió x las dudas?
        CSV.write(data.main_folder*"time_report.txt", time_file, delim='\t')

        #*********generar width report con std !
    end
end
