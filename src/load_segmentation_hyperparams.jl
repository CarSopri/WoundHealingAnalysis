
using TOML

Base.@kwdef mutable struct ParametersForAnalysis

    ###
    # Colors to use to highlight the segmentation done ***capaz debería tener nombre de fondo+izq+der no más ?
    ###
    black = RGB(0.0,0.0,0.0)
    red = RGB(1.0,0.0,0.0)
    green = RGB(0.0,1.0,0.0)

    number_of_ROIs::Int = 1
    L_for_roughness::Int = 500 # if ==0, roughness is calculated for the whole image. if bigger than the image, it is calculated over image's size **de dónde sale este nro ???
    n_of_Ls::Int = 20

    plot_hist::Bool = false
    plot_L::Bool = true

    save_folder::String = ""
    exclude::Vector{String} = []

end

Base.@kwdef mutable struct SegmentationHyperparameters

    # Felzenszwalb hyperparameters
    felzenszwalb_k::Int64 = 150000
    felzenszwalb_min_size::Int64 = 75000

    # Number of attempts to try before declaring segmentation failed
    k::Int64 = 0
    max_tries::Int64 = 15
    inv_max_tries::Float64 = 1/max_tries

    # Color threshold (in Grayscale) for eliminating possible shadows in the image
    shadow_threshold_color::Float64 = 0.25 # Pixels with a value below this will be considered shadow
    shadow_segment_threshold_color::Float64 = 0.6 # Segmentation results in a "bleaching" of the shadow, so this number should be higher (¨***creo q los convierto a blanco, es por eso en realidad no?)
    wound_threshold::Float64 = 0.7 # As a factor for mean image intensity

    # In px
    min_bp_filter::Int64 = 3
    max_bp_filter::Int64 = 40

    gaussian_blur_sigma::Int64 = 5

    # Check if segment flagged as wound is correct
    #image_partition::Int64 = 3 # esto divide la imagen en x partes iguales para ver que la sombra no esté tapando el centro de la imagen, en cuyo caso no se puede rescatar ninguna sección para segmentar
    margin::Int64 = 40 # esto hace que se corte un poquito más allá de la sombra encontrada por las dudas (en px)
    #band = (0.2,0.5,0.8) # As fraction of image's width; to check if segment found was uncentered and/or too close to the border, so it won't be flagged as the wound
    max_left_wall_wound_position::Float64 = 0.33
    wound_centre::Float64 = 0.5
    max_right_wall_wound_position::Float64 = 0.66
    # **creo que esto en realidad tendría que ser un chequeo de que la herida esté aprox adentro de la segmentación del tiempo anterior

    # Check if segment is too big to be the wound
    max_area_fraction::Float64 = 0.5 # As a fraction of total image area h*w
    # Check if maybe a second segment might also be a part of the wound
    merge_distance::Int64 = 300 # la distancia en px a la cual considerar que dos segmentos son una parte contigua de la misma herida (en px)
end

function load_data_from_hyperparams_file(hyperparams_file)

    hp_data = TOML.parsefile(hyperparams_file)
    hyperparams = SegmentationHyperparameters()

    #***agregar que si >1 o <0 salte error
    hyperparams.shadow_threshold_color = hp_data["Thresholding"]["shadow_threshold_color"]
    hyperparams.shadow_segment_threshold_color = hp_data["Thresholding"]["shadow_segment_threshold_color"]
    hyperparams.wound_threshold = hp_data["Thresholding"]["wound_threshold"]

    hyperparams.min_bp_filter = hp_data["BandPass"]["min_bp_filter"]
    hyperparams.max_bp_filter = hp_data["BandPass"]["max_bp_filter"]

    hyperparams.gaussian_blur_sigma = hp_data["Blur"]["gaussian_blur_sigma"]

    hyperparams.felzenszwalb_k = hp_data["Felzenszwalb"]["felzenszwalb_k"]
    hyperparams.felzenszwalb_min_size = hp_data["Felzenszwalb"]["felzenszwalb_min_size"]
    #***debería tener un valor máximo para que no pase a tener pruebas con params negativos !!
    hyperparams.max_tries = hp_data["Felzenszwalb"]["max_tries"]
    hyperparams.inv_max_tries = 1/hyperparams.max_tries

    #***agregar que si <0 salte error
    hyperparams.margin = hp_data["ProbableWoundPosition"]["margin"]
    #***agregar que si >1 o <0 salte error
    hyperparams.max_left_wall_wound_position = hp_data["ProbableWoundPosition"]["max_left_wall_wound_position"]
    hyperparams.wound_centre = hp_data["ProbableWoundPosition"]["wound_centre"]
    hyperparams.max_right_wall_wound_position = hp_data["ProbableWoundPosition"]["max_right_wall_wound_position"]
    hyperparams.max_area_fraction = hp_data["ProbableWoundPosition"]["max_area_fraction"]
    #***entre 0 y L tiene que ser!
    hyperparams.merge_distance = hp_data["ProbableWoundPosition"]["merge_distance"]

    return hyperparams
end

function save_hyperparams_to_file(hp::SegmentationHyperparameters, hyperparams_file::AbstractString)
    open(hyperparams_file, "w") do io
        println(io, "[Thresholding]")
        println(io, "shadow_threshold_color = ", hp.shadow_threshold_color)
        println(io, "shadow_segment_threshold_color = ", hp.shadow_segment_threshold_color)
        println(io, "wound_threshold = ", hp.wound_threshold)

        println(io, "\n[BandPass]")
        println(io, "min_bp_filter = ", hp.min_bp_filter)
        println(io, "max_bp_filter = ", hp.max_bp_filter)

        println(io, "\n[Blur]")
        println(io, "gaussian_blur_sigma = ", hp.gaussian_blur_sigma)

        println(io, "\n[Felzenszwalb]")
        println(io, "felzenszwalb_k = ", hp.felzenszwalb_k)
        println(io, "felzenszwalb_min_size = ", hp.felzenszwalb_min_size)
        println(io, "max_tries = ", hp.max_tries)

        println(io, "\n[ProbableWoundPosition]")
        println(io, "margin = ", hp.margin)
        println(io, "max_left_wall_wound_position = ", hp.max_left_wall_wound_position)
        println(io, "wound_centre = ", hp.wound_centre)
        println(io, "max_right_wall_wound_position = ", hp.max_right_wall_wound_position)
        println(io, "max_area_fraction = ", hp.max_area_fraction)
        println(io, "merge_distance = ", hp.merge_distance)
    end

    return nothing
end


function load_preprocessing_pipeline(path::AbstractString)
    config = TOML.parsefile(path)
    return String.(config["steps"])
end

function save_preprocessing_pipeline(steps, path::AbstractString)
    open(path, "w") do io
        TOML.print(io, Dict("steps" => String.(steps)))
    end
    return nothing
end

