code_folder = @__DIR__
include(joinpath(code_folder,"load_data.jl"))
include(joinpath(code_folder,"save_data.jl"))
include(joinpath(code_folder,"segment_image.jl"))

function initialize_report_files(data, report)
    if !report
        return 0, 0, 0
    end
    
    area_file = init_report_file(data.main_folder * "scratch_report.csv", data.time_series)
    width_file = init_report_file(data.main_folder * "width_report.csv", data.time_series)
    roughness_file = init_report_file(data.main_folder * "width_report.csv", data.time_series)
    time_file = init_report_file(data.main_folder * "time_report.csv", data.time_series)
    
    return area_file, width_file, roughness_file, time_file
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
function segment_one_image(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters, preprocessing_steps::Vector{String}, image_id::String, filepath::String, area_file::DataFrame, width_file::DataFrame, roughness_file::DataFrame, y_cut_all_t::Tuple{Union{Nothing,Int64},Union{Nothing,Int64}}, time::Int)
    save_images = data.save_images
    k = 0

    # First segmentation attempt.
    y_cut, keep_running = run_segmentation(
        data.main_folder,
        image_id,
        filepath,
        area_file,
        width_file,
        roughness_file,
        save_images,
        y_cut_all_t,
        segmentation_hyperparams,
        preprocessing_steps
    )


    if time == 1
        y_cut_all_t = y_cut

        # Second attempt when the first result indicates a cut was found.
        if y_cut != (1, 1960) #***acá debería pasarle el largo de la imagen
            y_cut, keep_running = run_segmentation(
                data.main_folder,
                image_id,
                filepath,
                area_file,
                width_file,
                roughness_file,
                save_images,
                y_cut_all_t,
                segmentation_hyperparams,
                preprocessing_steps
            )
            if y_cut == (-1, -1)
                return y_cut_all_t, true
            end

            #y_cut_all_t = y_cut
        end
    end

    # Retry if the segmentation has not yet been accepted.
    while keep_running && k < segmentation_hyperparams.max_tries
        y_cut, keep_running = run_segmentation(
            data.main_folder,
            image_id,
            filepath,
            area_file,
            width_file,
            roughness_file,
            save_images,
            y_cut_all_t,
            segmentation_hyperparams,
            preprocessing_steps
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


function save_segmentation_reports(data::DataInfo, area_file::DataFrame, width_file::DataFrame, roughness_file::DataFrame, time_file::DataFrame)
    CSV.write(
        joinpath(data.main_folder, "scratch_report.csv"),
        area_file;
        delim = '\t',
    )

    CSV.write(
        joinpath(data.main_folder, "width_report.csv"),
        width_file;
        delim = '\t',
    )

    CSV.write(
        joinpath(data.main_folder, "roughness_report.csv"),
        roughness_file;
        delim = '\t',
    )

    CSV.write(
        joinpath(data.main_folder, "time_report.csv"),
        time_file;
        delim = '\t',
    )

    return nothing
end

function roi_labels(rois::Union{String,Vector{String}})
    return rois isa String ? (isempty(rois) ? String[] : [rois]) : rois
end
function segment_all_files(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters, preprocessing_steps::Vector{String})
    validate_naming_convention(data)

    # Prepare reports and basic information.
    area_file, width_file, roughness_file, time_file = initialize_report_files(data, data.report)

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
                    ensure_report_row!(width_file, report_id, n_timepoints)
                    ensure_report_row!(roughness_file, report_id, n_timepoints)
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
                            preprocessing_steps,
                            image_id,
                            filepath,
                            area_file,
                            width_file,
                            roughness_file,
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
                    save_segmentation_reports(data, area_file, width_file, roughness_file, time_file)
                end
            end
        end
    end

    # Final report write, as in the original function.
    if data.report
        save_segmentation_reports(data, area_file, width_file, roughness_file, time_file)

        # TODO: Generate the width report with standard deviations.
    end

    return nothing
end


