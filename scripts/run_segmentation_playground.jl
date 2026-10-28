

function run_segmentation_playground(;image_path = nothing, hyperparams_path = nothing, preprocessing_path = nothing)
    project_dir = normpath(joinpath(@__DIR__, ".."))

    if image_path === nothing
        image_path = joinpath(project_dir, "examples", "example_image.tif")
    end

    image_dir = joinpath(splitpath(image_path)[1:end-1])
    image_filename = splitpath(image_path)[end]

    if hyperparams_path === nothing
        hyperparams_path = joinpath(project_dir, "configs", "Hyperparams.toml")
    end

    if preprocessing_path === nothing
        preprocessing_path = joinpath(project_dir, "configs", "PreprocessingSteps.toml")
    end

    for path in (image_path, hyperparams_path, preprocessing_path)
        isfile(path) || error("File not found: $path")
    end

    # Replace these calls with the actual functions already in your project.
    segmentation_hyperparams = load_data_from_hyperparams_file(hyperparams_path)
    steps = load_preprocessing_pipeline(preprocessing_path)

    segments, labels_segments, segmented_final_im = segmentation_steps(image_dir, image_filename, segmentation_hyperparams; steps=steps, display_steps = true)

end


