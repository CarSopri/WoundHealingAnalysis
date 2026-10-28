code_folder = @__DIR__
include(joinpath(code_folder,"save_data.jl"))
include(joinpath(code_folder,"segment_image_low_contrast.jl"))

function initialize_report_files(data, report)
    if !report
        return 0, 0
    end
    
    report_file = init_report_file(data.main_folder * "scratch_report.txt", data.time_batches)
    time_file = init_report_file(data.main_folder * "time_report.txt", data.time_batches)
    
    # Write initial empty files
    CSV.write(data.main_folder * "scratch_report.txt", report_file, delim='\t')
    CSV.write(data.main_folder * "time_report.txt", time_file, delim='\t')
    
    return report_file, time_file
end

function save_report_files(data, report_file, time_file)
    CSV.write(data.main_folder * "scratch_report.txt", report_file, delim='\t')
    CSV.write(data.main_folder * "time_report.txt", time_file, delim='\t')
end

function update_report_file_ids(report_file, time_file, pit, roi, total_time_frames)
    pit_roi_id = pit * string(roi)
    
    # Add to scratch report if not present
    if pit_roi_id ∉ report_file.ID
        push!(report_file, vcat([pit_roi_id], [0.0 for _ in 1:total_time_frames]))
    end
    
    # Add to time report if not present
    if pit_roi_id ∉ time_file.ID
        push!(time_file, vcat([pit_roi_id], [0.0 for _ in 1:total_time_frames]))
    end
end

function update_time_report(data, time_file, pit, roi, imfile, full_id)
    # Reference file is the first time point of the first batch #**acá también, ojo con el orden del namefile porque debería venir de config !!! T.T
    ref = data.main_folder * data.sub_folders[1] * 
          pit * string(roi) * data.name_time_batches[1] * 
          data.common_name * lpad(1, 3, "0") * data.file_format
    
    time_from_file = (mtime(imfile) - mtime(ref)) / 3600  # in hours
    
    # Find the correct row and column to update
    row = findfirst(time_file.ID .== full_id[1:end-12])
    col = parse(Int, full_id[end-2:end])
    
    time_file[row, col + 1] = time_from_file
end


function first_pass_collect_cuts(data, pit, roi, report_file, time_file, tf, save_images, segmentation_hyperparams, report)
    cuts_per_time = Dict{Int, Tuple{Int,Int}}()
    initial_cut = (nothing, nothing)  # Start with no cut

    # Update report files with new IDs if needed
    if report
        update_report_file_ids(report_file, time_file, pit, roi, tf)
    end
    
    for (batch_idx, time_batch) in enumerate(data.time_batches)
        general_id = pit * string(roi) * data.name_time_batches[batch_idx] #**ojo acá y en ref porque estoy cambiando el orden respecto del primer exp we
        
        for time in time_batch
            # Build file paths and IDs
            subfolder = data.sub_folders[batch_idx]
            frame_number = time - time_batch[1] + 1
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
                update_time_report(data, time_file, pit, roi, imfile, full_id)
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

function second_pass_refine_cuts(data, pit, roi, better_cut, old_cuts, 
                                  report_file, time_file, save_images, 
                                  segmentation_hyperparams, report)
    current_lower, current_upper = better_cut
    
    for (batch_idx, time_batch) in enumerate(data.time_batches)
        general_id = pit * string(roi) * data.name_time_batches[batch_idx]
        
        for time in time_batch
            # Only re-segment frames that actually need the larger cut
            old_cut = old_cuts[time]
            needs_larger_cut = old_cut[1] > current_lower || old_cut[2] < current_upper
            
            if needs_larger_cut
                # Build file paths
                frame_number = time - time_batch[1] + 1
                general_name = general_id * data.common_name * lpad(frame_number, 3, "0") * data.file_format
                imfile = data.main_folder * data.sub_folders[batch_idx] * general_name
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

function process_pit_roi_combination(data, pit, roi, report_file, time_file, tf, save_images, segmentation_hyperparams, report)
    # Store cuts found for each time point
    cuts_per_time = Dict{Int, Vector{Int}}()
    
    # First pass: collect all cuts without re-segmentation
    cuts_per_time = first_pass_collect_cuts(
        data, pit, roi, report_file, time_file, tf,
        save_images, segmentation_hyperparams, report
    )
    
    # Analyze if refinement is needed
    refinement_needed, better_cut = analyze_if_refinement_needed(cuts_per_time)
    
    # Second pass: re-segment if needed
    if refinement_needed
        @info "Refinement needed! Re-segmenting with better cut: $better_cut"
        second_pass_refine_cuts(
            data, pit, roi, better_cut, cuts_per_time,
            report_file, time_file, save_images, segmentation_hyperparams, report
        )
    end
    
    # Save after each pit/roi combination
    if report
        save_report_files(data, report_file, time_file)
    end
end
    

function run_segmentation_for_all_files_new(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters)
    report = data.report
    save_images = data.save_images

    # Get list of pits to process (excluding ignored ones)
    pits_to_process = [p for p in data.pit_names if p ∉ data.ignore_pits]
    tf = data.time_batches[end][end]

    ID = String
    general_name = String
    subfolder = String
    
    # Si no existe el file, crearlo, y si existe, hacer un df
    report_file, time_file = initialize_report_files(data, report)

    l = 0
    for pit in pits_to_process
        for roi in data.rois
            l+=1
            if l!=1 #****************** esto debería tener en cuenta el file_start o algo así que puse en config !!
                continue
            end

            process_pit_roi_combination(
                data, pit, roi, report_file, time_file, tf,
                save_images, segmentation_hyperparams, report
            )
            
        end
    end
    
    if report
        save_report_files(data, report_file, time_file)
    end
end

#*****esta sacarla
function run_segmentation_for_all_files(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters)
    report = data.report
    save_images = data.save_images
    
    pit = [p for p in data.pit_names if p ∉ data.ignore_pits]
    tf = data.time_batches[end][end]

    ID = String
    general_name = String
    subfolder = String
    
    # Si no existe el file, crearlo, y si existe, hacer un df
    if report
        report_file = init_report_file(data.main_folder*"scratch_report.txt", data.time_batches)
        time_file = init_report_file(data.main_folder*"time_report.txt", data.time_batches)
    else 
        report_file = 0
        time_file = 0
    end

    l = 0
    
    for p in pit
        for n_photo in data.rois
            l+=1
            #if l<13
            #    continue
            #end
    
            y_cut_all_t = (nothing,nothing)

            for (n_batch,time_batch) in enumerate(data.time_batches)

                general_ID = p*string(n_photo)*data.name_time_batches[n_batch] #**ojo acá y en ref porque estoy cambiando el orden respecto del primer exp we

                if report
                    
                    # Check if ID is already a row in report_file
                    #**acá debería ser p*string(n_photo); lo del medio debería cambiar sólo con t así que no hay que repetirlo en rows
                    if p*string(n_photo) ∉ report_file.ID #**acá ojooo es p para el primer exp
                        push!(report_file, vcat([p*string(n_photo)],[0.0 for i in 1:tf]))
                    end
                    if p*string(n_photo) ∉ time_file.ID 
                        push!(time_file, vcat([p*string(n_photo)],[0.0 for i in 1:tf]))
                    end
                end
                
                for time in time_batch
                    
                    #time_displacement = n_batch!=1 ? time_batch[n_batch-1][end] : 0
                    subfolder = data.sub_folders[n_batch]
                    general_name = general_ID*data.common_name*lpad(time-time_batch[1]+1,3,"0")*data.file_format
                    ID = general_ID*"_T"*lpad(time,3,"0")

                    if !isfile(data.main_folder*subfolder*general_name)
                        @info (data.main_folder*subfolder*general_name, "\n!!!!!!!!!!!!!!!!!!!!!! FILE NOT FOUND !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        continue
                    end

                    @info ("#####################   ",ID,"   #####################")

                    if report
                        ref = data.main_folder*data.sub_folders[1]*p*string(n_photo)*data.name_time_batches[1]*data.common_name*lpad(1,3,"0")*data.file_format
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
                    y_cut, keep_running = run_segmentation(data.main_folder, ID, data.main_folder*subfolder*general_name, report_file, save_images,y_cut_all_t, segmentation_hyperparams)
                    
                    if time == 1
                        y_cut_all_t = y_cut
                    end
                    # Esperaría que en el segundo paso no recorte más xq ya está, pero que segmente mejor
                    # Si no había lente presente, este paso es redundante, sólo repite el anterior
                    if y_cut != (1,1960)
                        y_cut, keep_running = run_segmentation(data.main_folder, ID, data.main_folder*subfolder*general_name, report_file, save_images, y_cut_all_t, segmentation_hyperparams)
                        if y_cut == (-1,-1)
                            break
                        end
                        if time == 1
                            y_cut_all_t = y_cut
                        end
                    end
                    # In case the segmentation wasn't right
                    while keep_running && k<15 #***debería ser un input parameter
                        y_cut, keep_running = run_segmentation(data.main_folder, ID, data.main_folder*subfolder*general_name, report_file, save_images, y_cut_all_t, segmentation_hyperparams)
                        k+=1
                        @info ("-----> Number of iterations: ",k, ", Cut in y (if shadow present): ",y_cut)
                        if time == 1
                            y_cut_all_t = y_cut
                        end
                    end
                end
            end

            if report
                CSV.write(data.main_folder*"scratch_report.txt", report_file, delim='\t') #***esto lo tendría que adaptar para que vaya escribiendo cada tanto aunque sea idk; capaz un log que anote de dónde hasta dónde corrió x las dudas?
                CSV.write(data.main_folder*"time_report.txt", time_file, delim='\t')
            end
            
        end
    end
    
    if report
        CSV.write(data.main_folder*"scratch_report.txt", report_file, delim='\t') #***esto lo tendría que adaptar para que vaya escribiendo cada tanto aunque sea idk; capaz un log que anote de dónde hasta dónde corrió x las dudas?
        CSV.write(data.main_folder*"time_report.txt", time_file, delim='\t')

        #*********generar width report con std !
    end
end
