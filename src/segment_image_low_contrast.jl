
code_folder = @__DIR__
include(joinpath(code_folder,"image_analyzers.jl"))
#**otra cosa que tengo que revisar es el ignore_pits... me da un poco de fiaca que sea algo que haya que cargar manualmente. creo que debería poder ignorar automáticamente esos archivos si no existen, y cargarlos
# capaz en un .txt de ignore_pits para que busque ahí directamente los que ignoró y se los saltee en vez de intentar cargar todos cada vez

#**tengo que revisar también el renaming de la convención de los primeros fibroblastos respecto del resto de datos ....

#***************recordá que k debería ser k de segmentation_hyperparams !!! bah o capaz no, porque es mutable a diferencia de todo lo demás !

#***chat gpt dice que printear es muy lento, así que dejarlo al mínimo para hacer seguimiento 
#***also chequear bien los hiperparámetros que faltan en segmentation_steps. ver cuáles vale la pena tunear y tratar de explicarlos bien
# y chequear qué cambio respecto de la segmentación de segment_image ?
# y bueno ver bien los logs que saca y cuáles son relevantes
#***no me gusta nada que en funciones que nada que ver tnego que pasar y_cut y keep_running sólo porque necesito cortar en pasos intermedios
# ***debería pensar un return alternativo o alguna manera de flaggear estos casos idk xq siento que así como está ahora no se entiende nada

function segmentation_steps(im, segmentation_hyperparams)

    #***poner esto en test_segmentation2 y ver cómo se diferencia i guess.
    #***creo incluso que debería ser una función en test_segmentation que se importe acá?
    gray_im = Gray.(im)

    # Locate lens shadow (if present)
    lens_shadow_indices, min_h, max_h, min_w, max_w = isolate_lens_shadow(gray_im, segmentation_hyperparams.shadow_threshold_color)
    

    bp = Gray.(1.0.-(bandpass_filter(gray_im, 3, 40))) #1.0 .- #***qué es ese 3,40?? deberían ser hyperparameters? !!!!!!!!!!!!!!!!!!

    #corrected_img = Gray.(fiji_subtract_background(bp,50, true)./255)
    #corrected_img = Gray.(fiji_subtract_background_visually(bp;radius=50, light_background=true, smoothing=true))
    #corrected_img = Gray.(rolling_ball_background(bp; radius=50, light_background=true, presmooth=true, output_type=UInt8))
    #corrected_img = Gray.(fiji_subtract_background_exact(bp;radius=50, light_background=true, disable_smoothing=false)) 
    
    #corrected_img = Gray.(fiji_subtract_background_merged(bp; radius=50, light_background=true, smoothing=true)./255)

    ##############************ me estoy salteando corrected_img
    equalized = adjust_histogram(bp, Equalization(nbins = 256)) #**no me acuerdo qué significan esos deciles ni qué estoy printeando !
    
    # Detect edges with Laplacian Kernel
    filtered_gray_im = detect_edges(equalized)
    
    # Blur the image
    discrete_filtered_gray_im = blur_image!(filtered_gray_im,5,1) # **por qué los pixeles blancos no vuelven blancos ?? **ese 5 debería ser hyperparam
    
    corrected_img = Gray.(threshold_colors.(discrete_filtered_gray_im,mean(filtered_gray_im)*0.6)) #**ese 0.6 debería ser hyperparam
    
    # Separate lens shadow (if present) # **podría restar los 1.0 que no quiero ?
    corrected_img[lens_shadow_indices] .= 1.0
    
    discrete_final_im = corrected_img
    
    # Get image segmentation
    segments = felzenszwalb(discrete_final_im, segmentation_hyperparams.meansize_wound, segmentation_hyperparams.minsize_wound) #imagen, tamaño aprox de clusters, mínimos píxeles para hacer sección
    labels_segments = labels_map(segments)
    segmented_final_im = map(i->segment_mean(segments,i), labels_segments) # segment_mean busca el color promedio del segmento y lo asigna a todo el segmento

    return segments, labels_segments, segmented_final_im
end


# Display image steps **agregar títulos o labels ?
    #display([im filtered_gray_im
    #discrete_filtered_gray_im corrected_img
    #discrete_final_im segmented_final_im ])

function felzenszwalb_hyperparams(k, meansize, minsize, max_tries)
    """
    Run to try smaller and smaller segments while segmentation tries keep failing. Stop after max_tries attempts.
    """
    
    #meansize_segmentation = 150000 - k*10000 #*** esto probablemente debería depender del tamaño de la imagen? y del tamaño relativo que esperás que tenga la herida. capaz debería ser un %
    #minsize_segmentation = 75000 - k*5000

    meansize_segmentation = meansize - k*(meansize/max_tries) #** guardar 1/max_tries que no va a cambiar en el loop
    minsize_segmentation = minsize - k*(minsize/max_tries) 
    
    return meansize_segmentation, minsize_segmentation
end

function cut_big_shadows(im_file, y_cut)

    # Load image and turn to Gray
    im = load(im_file)
    (h,w) = size(im)
    #@info ("cut: ",y_cut)

    # The cut was flagged in a previous segmentation as a big section of the image that is mostly dark (shadow)
    if y_cut != (nothing,nothing)        
        im = im[y_cut[1]:y_cut[2], 1:end]
        (h,w) = size(im)
    end

    # If most of the image is shadow, then no segmentation will be possible "***creo que esto lo puse afuera en first_pass_collect_cuts de segmentation_pipeline.jl
    if h==0 || w ==0
        @info ("!!!!!!!!!!!!!!!! No segmentation was possible. Seems like the whole image is in shadow. !!!!!!!!!!!!!!!!")
        return im, h, w, (-1,-1), false
    end
    #@info ("Image height and length: ",h, ", ",w)
    return im, h, w, y_cut, true
end


function cut_at_top_and_bottom(segmented_final_im, value_shadow, image_partition, h)
    y_23h = highest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13h = highest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    y_23l = lowest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13l = lowest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    #@info ("Where to cut: top-right corner: ",y_23h," top-left corner: ",y_13h," bottom-right: ",y_23l," bottom-left: ",y_13l)

    if y_23h == h || y_13h == h
        if y_13l != nothing && (y_23l == nothing || y_23l>y_13l)
            y_cut2 = y_13l
        elseif y_23l != nothing && (y_13l == nothing || y_13l>y_23l) 
            y_cut2 = y_23l
        else
            return (nothing,nothing)
        end
        y_cut1 = 2
    elseif y_23l == 1 || y_13l == 1
        if y_23h != nothing && (y_13h == nothing || y_23h>y_13h)
            y_cut1 = y_23h
        elseif y_13h != nothing && (y_23h == nothing || y_13h>y_23h) 
            y_cut1 = y_13h
        else
            return (nothing,nothing)
        end
        y_cut2 = h-1
    end
    return (y_cut1,y_cut2)
end
function cut_at_top(segmented_final_im, value_shadow, image_partition, bm, lm, rm, h, w, margin)
    y_cut1 = bm[1]
    y_cut2 = h-1
    y_23 = highest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow) 
    y_13 = highest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    #@info ("Where to cut in y: if lens shadow is on the right: ",y_23," if it's on the left: ",y_13)
    if y_23 != nothing && (y_13 == nothing || y_23>y_13) && image_partition[2]<bm[2]
        y_cut1 = y_23
    elseif y_13 != nothing && (y_23 == nothing || y_13>y_23) && image_partition[2]>bm[2]
        y_cut1 = y_13
    elseif y_23 == nothing && y_13 == nothing && (lm[2]==0 || rm[2]==w)
        y_cut1 = 1
    end

    if y_cut1 < h-margin #**me acabo de dar cuenta de que si no hay corte igual corto ese margen mmmm
        y_cut1 = y_cut1+margin
    end
    return (y_cut1,y_cut2)
end
function cut_at_bottom(segmented_final_im, value_shadow, image_partition, tm, lm, rm, h, w, margin)
    y_cut1 = 2
    y_cut2 = tm[1]
    y_23 = lowest_y_for_x_and_value(segmented_final_im, image_partition[2], value_shadow)
    y_13 = lowest_y_for_x_and_value(segmented_final_im, image_partition[1], value_shadow)
    if y_23 != nothing && (y_13 == nothing || y_23<y_13) && image_partition[2]<tm[2]
        y_cut2 = y_23
    elseif y_13 != nothing && (y_23 == nothing || y_13<y_23) && image_partition[2]>tm[2]
        y_cut2 = y_13
    elseif y_23 == nothing && y_13 == nothing && (lm[2]==0 || rm[2]==w)
        y_cut2 = h
    end
    if y_cut2 > margin
        y_cut2 = y_cut2-margin
    end
    return (y_cut1,y_cut2)
end


function identify_and_cut_shadow(h, w, y_cut, segments, segmented_final_im, shadow_segment_threshold_color, margin, fraction_of_image)

    # Calculate area of shadow if present and cut it out of the image in T001
    value_shadow = maximum(values(segment_mean(segments)))
    #@info ("Shadow color: ",value_shadow)

    shadow_present = value_shadow >=shadow_segment_threshold_color

    if !shadow_present || y_cut != (nothing,nothing)# y_cut == (nothing,nothing)only on first timeframe
        #@info ("No cut: ",y_cut)
        return (1,h), false
    else
        image_partition = (w÷fraction_of_image,w-w÷fraction_of_image)

        lm,rm = find_leftmost_and_rightmost(segmented_final_im,value_shadow) # index 1 is vertical, 2 is horizontal
        tm,bm = find_topmost_and_bottommost(segmented_final_im,value_shadow)
        #@info ("Shadow position: leftmost px: ",lm," rightmost px: ",rm," topmost px: ",tm," bottommost px: ",bm)

        y_cut1 = 2
        y_cut2 = h-1

        if tm[1] == 1 && bm[1] == h 
            y_cut1, y_cut2 = cut_at_top_and_bottom(
                segmented_final_im, value_shadow, image_partition, h
            )

        elseif tm[1] == 1
            y_cut1, y_cut2 = cut_at_top(
                segmented_final_im, value_shadow, image_partition, bm, lm, rm, h, w, margin
            )

        elseif bm[1] == h
            y_cut1, y_cut2 = cut_at_bottom(
                segmented_final_im, value_shadow, image_partition, tm, lm, rm, h, w, margin
            )
        end


        @info ("Where to cut: ",(y_cut1,y_cut2))
        return (y_cut1,y_cut2), true
    end

end


function find_probable_wound_segment(w, segmented_final_im, segments, value_scratch, area_scratch_index, band, k, max_tries)

    # The band is to check if the wound found is away from the center or if it's too close to the border of the image
    left_band = band[1]*w
    central_band = band[2]*w
    right_band = band[3]*w
    
    lm1,rm1 = find_leftmost_and_rightmost(segmented_final_im,value_scratch)
    
    # If biggest darkest area is far away from the center, it might not be the wound
    if rm1[2]<central_band || lm1[2]>central_band 

        # We loop over every segment in case one of them might be the wound
        for s in segment_labels(segments) 
            #println("Try other segments: ",s," Color value: ",segment_mean(segments,s))
            lms,rms = find_leftmost_and_rightmost(segmented_final_im,segment_mean(segments,s))

            # If biggest darkest area is very close to the border of the image, it might not be the wound
            if lms[2]> left_band && rms[2] < right_band       
                #println("Leftmost and rightmost px values for this segment: ",lms[2],"  ",rms[2])
                value_scratch = segment_mean(segments,s)
                area_scratch_index = s
                lm1 = lms
                rm1 = rms
                return true
            end
        end
        
        if k == (max_tries-1)  
            @info ("!!!!!!!!!!!!!!!! The last try didn't work, no wound found !!!!!!!!!!!!!!!!")
            return false
        else
            @info ("COULDN'T FIND WOUND. Redoing steps for smaller segmentation hyperparameters.")
            return true
        end
        
    else
        return true
    end
end


function is_wound_bigger_than_max_area(max_area,area_scratch, k, max_tries)

    # If the segmentation was not right:
    if area_scratch >= max_area 
        if k == (max_tries-1)
            #y_cut = (1,h) #** me sirve devolverlo así ?
            @info ("!!!!!!!!!!!!!!!! The last try didn't work, segmentation failed !!!!!!!!!!!!!!!!") #@warn
            return true, false # Si el último es false, deja de loopear, si es true, sigue intentando segmentar con otros hyperparams. Si el segundo es false, la segmentación anduvo bien
        else
            @info ("COULDN'T DO SEGMENTATION. Redoing steps for smaller segmentation hyperparameters.")
            return true, true
        end
    else 
        return false, false
    end
end

function count_label(idx, labels)
    c = 0
    @inbounds for x in labels
        c += (x == idx)
    end
    return c
end

@inline function spans_are_close(lm1, rm1, lm2, rm2, distance)
    abs(lm2[2] - lm1[2]) ≤ distance &&
    abs(rm2[2] - rm1[2]) ≤ distance
end

function find_second_darkest_area(segmented_final_im, segments, labels_segments, value_scratch, area_scratch, area_scratch_index, merge_distance)  
    lm1,rm1 = find_leftmost_and_rightmost(segmented_final_im,value_scratch) #################
    
    # In case the area is found but a part has already closed
    value_scratch2 =  minimum([v for (k,v) in segment_mean(segments) if k!= area_scratch_index]) 
    
    lm2,rm2 = find_leftmost_and_rightmost(segmented_final_im,value_scratch2)

    if spans_are_close(lm1,rm1,lm2,rm2,merge_distance) #(lm2[2] <= lm1[2]+merge_distance) && (lm2[2] >= lm1[2]-merge_distance) && (rm2[2] <= rm1[2]+merge_distance) && (rm2[2] >= rm1[2]-merge_distance)
        area_scratch_index2 = find_biggest_dark_area(segments, value_scratch2, area_scratch_index)
        area_scratch = area_scratch + count_label(area_scratch_index2, labels_segments)
        @info ("First area: ",area_scratch_index, ", Second area: ", area_scratch_index2)
        
        return area_scratch, area_scratch_index2, true
    else
        return area_scratch, 0, false
    end
    #end
    
end


function compute_edges!(edges::Matrix{UInt8},labels::Matrix{Int},wound_idx::Int,wound_idx2::Union{Int,Nothing})
    h, w = size(labels)

    @inbounds for i in 2:h-1, j in 2:w-1
        seg = labels[i,j]

        up    = labels[i-1,j]
        down  = labels[i+1,j]
        left  = labels[i,j-1]
        right = labels[i,j+1]

        # helper
        is_wound(x) = x == wound_idx || x == wound_idx2

        if up != seg
            edges[i,j] = is_wound(up) ? 2 : 1
        elseif down != seg
            edges[i,j] = is_wound(down) ? 2 : 1
        elseif left != seg
            edges[i,j] = is_wound(left) ? 2 : 1
        elseif right != seg
            edges[i,j] = is_wound(right) ? 2 : 1
        end
    end

    #@info ("COUNT ",count(x->x==1,edges),"  ",count(x->x==2,edges)) #**esto para tener en cuenta si un lado es mayor que el otro? pero podría ser por el corte de la lente
    return edges
end

function paint_edges(im, edges)
    out = RGB.(im)  # one allocation

    @inbounds for idx in eachindex(edges)
        e = edges[idx]
        if e == 2
            out[idx] = RGB(1,0,0)
        elseif e == 1
            out[idx] = RGB(0,0,1)
        end
    end

    return out
end


function classify_segments!(
    edge_img,
    segmented_img,
    segments,
    labels,
    wound_idx,
    wound_idx2,
    w,
    mean_y
)
    for seg in segment_labels(segments)
        if seg == wound_idx || seg == wound_idx2
            paint_segment!(edge_img, labels, seg, RGB(0,0,0))
            continue
        end

        lm, rm = find_leftmost_and_rightmost(segmented_img, segment_mean(segments,seg))

        if lm[2] == 1 && rm[2] != w
            paint_segment!(edge_img, labels, seg, RGB(1,0,0))

        elseif lm[2] != 1 && rm[2] == w
            paint_segment!(edge_img, labels, seg, RGB(0,1,0))

        else
            split_segment!(edge_img, labels, seg, mean_y)
        end
    end
end

function paint_segment!(img, labels, seg, color)
    @inbounds for i in eachindex(labels)
        if labels[i] == seg
            img[i] = color
        end
    end
end

function split_segment!(img, labels, seg, mean_y)
    h,w = size(labels)
    @inbounds for i in 1:h, j in 1:w
        if labels[i,j] == seg
            img[i,j] = j < mean_y ? RGB(1,0,0) : RGB(0,1,0)
        end
    end
end
    




function run_segmentation(main_folder, ID, im_file, report_file, save_images, y_cut, segmentation_hyperparams)
    """
    report_file = 0 if you don't want to save areas in a .txt; else, should be the string with the name of the file
    """
    
    """
    Params for segmentation: an external loop runs this segmentation code and keeps making the felzenszwalb parameters smaller any time the segmentation fails
    """
    meansize_segmentation, minsize_segmentation = felzenszwalb_hyperparams(segmentation_hyperparams.k, segmentation_hyperparams.meansize_wound, segmentation_hyperparams.minsize_wound, segmentation_hyperparams.max_tries)

    """
    If y_cut is not [nothing,nothing], cut the image where shadows were previously found.
    """
    im, h, w, y_cut, keep_running = cut_big_shadows(im_file, y_cut)
    # If a cut was needed to remove shadows from the image, the segmentation stops here and is run again on the cut image in an external loop. 
    if !keep_running
        return y_cut, keep_running
    end

    """
    Run preprocessing of the image and segmentation algorithm.
    """
    segments, labels_segments, segmented_final_im = segmentation_steps(im, segmentation_hyperparams)

    """
    Find if there are any shadows present in the image (this is analyzed from the segmented image), and escape the function to redo segmentation if a cut is needed.
    """
    #***esto debería tener una lógica distinta porque sólo va a correr en el primer paso pero no queda super explícito como está ahora (y_cut=(nothing,nothing) sólo en T=001)
    y_cut, cut_made = identify_and_cut_shadow(h,w, y_cut, segments, segmented_final_im, segmentation_hyperparams.shadow_segment_threshold_color, segmentation_hyperparams.margin, segmentation_hyperparams.image_partition)
    # Run segmentation again if a shadow was identified, now with the image cut where the shadow ends.
    if cut_made
        return y_cut,cut_made 
    end

    @info ("Number of segments detected: ", length(segment_labels(segments)))


    # Calculate wound area
    value_scratch = minimum(values(segment_mean(segments)))
    area_scratch_index = find_biggest_dark_area(segments, value_scratch)
    area_scratch = count_label(area_scratch_index, labels_segments)

    # Post-processing
    found_wound = find_probable_wound_segment(w, segmented_final_im, segments, value_scratch, area_scratch_index, segmentation_hyperparams.band, segmentation_hyperparams.k, segmentation_hyperparams.max_tries)

    if !found_wound
        return y_cut, found_wound
    end

    max_area = segmentation_hyperparams.max_area_fraction*w*h
    two_areas = false

    # Maybe the biggest darkest area is not the real wound
    #println(value_scratch)    

    area_is_bigger, keep_running = is_wound_bigger_than_max_area(max_area,area_scratch, segmentation_hyperparams.k, segmentation_hyperparams.max_tries)
    if area_is_bigger
        return y_cut,keep_running
    end

    area_scratch, area_scratch_index2, two_areas = find_second_darkest_area(segmented_final_im, segments, labels_segments, value_scratch, area_scratch, area_scratch_index, segmentation_hyperparams.merge_distance)

    

    edges = zeros(UInt8,h,w)
    compute_edges!(edges, labels_segments, area_scratch_index, area_scratch_index2) 
    edges_image = paint_edges(im, edges)
    #wound_mask = build_wound_mask(labels_segments, h, w, area_scratch, area_scratch_index2)
    wound_edge_img = RGB.(segmented_final_im) # fill(Gray(1.0), h, w)
    mean_y = get_mean_vertical_axis(labels_segments, h,w, area_scratch_index)
    classify_segments!(wound_edge_img, segmented_final_im, segments, labels_segments, area_scratch_index, area_scratch_index2, w, mean_y)

    
    if save_images
        # Save segmented image
        #save(main_folder*"segmented_image_"*ID*".png",colorview(Gray,segmented_final_im))
        # Save original image with edges from segments drawn on top
        save(main_folder*"image_"*ID*".jpg",edges_image)
        # Save wound image with edges
        save(main_folder*"wound_edges_image_"*ID*".png",wound_edge_img)
    end

    if report_file != 0
        #**cuidado si la fila no existe la tengo que agregar, y idem la columna (pero capaz eso desde afuera)
        row = findfirst(report_file.ID .== ID[1:end-12]) # pit name
        col = parse(Int64,ID[end-2:end]) # time value
        #@info ("Row: ",row," Col: ",col)

        report_file[row, col+1] = area_scratch
        #print(report_file,"\t",length(findall(item -> item == area_scratch, labels_segments)))
    end
    #@info ("Area for ",ID,": ",area_scratch)

    return y_cut, false
end

