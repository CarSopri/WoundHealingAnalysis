
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
    # ***este exclude gigante es una bazofia, debería estar automatizado. debería guardar en un exclude.txt las que salteo cuando hago la segmentación ? y un log que explique por qué
    # y tendría que irse actualizando a medida que se corre de nuevo, sumando líneas que no estaban y chequeando si se acomodó una imagen que estaba flageada.
    exclude::Vector{String} = [] #***esto debería ir llenándose con ignore_pits y después cargarlo en un .txt que, si existe, lo levante directamente, y se puedan agregar datos nuevos si hace falta.
            #"A2_03_1_1","A2_03_1_2","A3_03_1_1","A3_03_1_3","A4_03_1_1","A5_03_1_1","A5_03_1_3","A6_03_1_1",
            #"B1_03_1_1","B1_03_1_2","B3_03_1_1","B3_03_1_2","B3_03_1_4","B4_03_1_4","B5_03_1_1","B5_03_1_5",
            #"C1_03_1_3","C4_03_1_5","C5_03_1_1","C5_03_1_5",
            #"D2_03_1_2","D2_03_1_3","D3_03_1_1","D3_03_1_3","D4_03_1_2","D4_03_1_4","D4_03_1_5","D5_03_1_2","D5_03_1_3","D5_03_1_4","D6_03_1_2","D6_03_1_3","D6_03_1_4"]

end

Base.@kwdef mutable struct SegmentationHyperparameters #*** estos también debería cargarlos en config.toml

    # Felzenszwalb hyperparameters
    meansize_wound::Int64 = 150000
    minsize_wound::Int64 = 75000

    # Number of attempts to try before declaring segmentation failed
    k::Int64 = 0
    max_tries::Int64 = 15
    inv_max_tries::Float64 = 1/max_tries

    # Color threshold (in Grayscale) for eliminating possible shadows in the image
    shadow_threshold_color::Float64 = 0.25 # Pixels with a value below this will be considered shadow
    shadow_segment_threshold_color::Float64 = 0.6 # Segmentation results in a "bleaching" of the shadow, so this number should be higher (¨***creo q los convierto a blanco, es por eso en realidad no?)
    wound_threshold::Float64 = 0.6 # As a factor for mean image intensity

    # In px
    min_bp_filter::Int64 = 3
    max_bp_filter::Int64 = 4

    gaussian_blur_sigma::Int64 = 5

    # Check if segment flagged as wound is correct
    #image_partition::Int64 = 3 # esto divide la imagen en x partes iguales para ver que la sombra no esté tapando el centro de la imagen, en cuyo caso no se puede rescatar ninguna sección para segmentar
    margin::Int64 = 40 # esto hace que se corte un poquito más allá de la sombra encontrada por las dudas (en px)
    #band = (0.2,0.5,0.8) # As fraction of image's width; to check if segment found was uncentered and/or too close to the border, so it won't be flagged as the wound
    max_left_wall_wound_position::Float64 = 0.2
    wound_centre::Float64 = 0.5
    max_right_wall_wound_position::Float64 = 0.8
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

    hyperparams.meansize_wound = hp_data["Felzenszwalb"]["meansize_wound"]
    hyperparams.minsize_wound = hp_data["Felzenszwalb"]["minsize_wound"]
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

