
project_dir = normpath(joinpath(@__DIR__),"..")

include(joinpath(project_dir,"src","segmentation_pipeline.jl"))
include(joinpath(project_dir,"src","load_data.jl"))

config_file = joinpath(project_dir,"examples","example_experiment1","config_exp1.toml")
hyperparams_file = joinpath(project_dir, "examples","example_experiment1", "Hyperparams.toml")
preprocessing_steps_file = joinpath(project_dir, "examples","example_experiment1", "PreprocessingSteps.toml")

data = load_data_from_config_file(config_file)
segmentation_hyperparams = load_data_from_hyperparams_file(hyperparams_file)
preprocessing_steps = load_preprocessing_pipeline(preprocessing_steps_file)

function time_segmentation_per_file(data::DataInfo, segmentation_hyperparams::SegmentationHyperparameters, preprocessing_steps::Vector{String})
    validate_naming_convention(data)

    # Prepare reports and basic information.
    area_file, width_file, roughness_file, time_file = initialize_report_files(data, data.report)

    n_timepoints = data.time_series[end][end]
    rois = roi_labels(data.rois)
    println("rois ",rois)

    # Process each channel, well, ROI, and time series.
    for channel in data.channels
        for (well_number,well) in enumerate(data.wells_used)
            for (roi_number,roi) in enumerate(rois)
                if !isempty(data.keep_rois) && data.keep_rois[well_number]<roi_number
                    continue
                end

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

                        start_time = time_ns()

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

                        elapsed = (time_ns() - start_time) / 1e9

                        println("$(filepath): $(round(elapsed, digits=3)) s")

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

    end

    return nothing
end

time_segmentation_per_file(data, segmentation_hyperparams, preprocessing_steps)



