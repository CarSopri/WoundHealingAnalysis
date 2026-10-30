



function find_leftmost_and_rightmost(mat, value)
    zero_indices = findall(x -> x == value, mat)  # Find all zeros in the matrix

    if isempty(zero_indices)
        return nothing, nothing  # If no zeros, return nothing
    end

    # Find leftmost (smallest column index)
    leftmost_zero = argmin(x -> x[2], zero_indices)

    # Find rightmost (largest column index)
    rightmost_zero = argmax(x -> x[2], zero_indices)

    return leftmost_zero, rightmost_zero
end

function find_topmost_and_bottommost(mat,value)
    zero_indices = findall(x -> x == value, mat)  # Find all zeros in the matrix

    if isempty(zero_indices)
        return nothing, nothing  # If no zeros, return nothing
    end

    # Find topmost (smallest row index)
    topmost_zero = argmin(x -> x[1], zero_indices)

    # Find bottommost (largest row index)
    bottommost_zero = argmax(x -> x[1], zero_indices)

    return topmost_zero, bottommost_zero
end

function lowest_y_for_x_and_value(mat, x_target, val)
    rows = size(mat, 1)
    ys = []

    for y in 1:rows
        if mat[y, x_target] == val
            push!(ys, y)
        end
    end

    return isempty(ys) ? nothing : minimum(ys)
end

function highest_y_for_x_and_value(mat, x_target, val)
    rows = size(mat, 1)
    ys = []

    for y in 1:rows
        if mat[y, x_target] == val
            push!(ys, y)
        end
    end

    return isempty(ys) ? nothing : maximum(ys)
end

function get_mean_vertical_axis(labels_im, h,w, segment_index)
    count_y = 0
    count_k = 0
    for row in 1:h
        for y in 1:w
            if labels_im[row,y] == segment_index
                count_y += y
                count_k +=1
            end
        end
    end
    return count_y/count_k
end

function get_segment_stats(labels_im, h,w, segment_index)
    h, w = size(labels_im)

    # Mean x-coordinate over all pixels in the segment
    sum_x = 0
    n_pixels = 0

    # Online statistics for the width distribution across rows
    n_rows = 0
    mean_width = 0.0
    M2_width = 0.0

    @inbounds for row in 1:h
        xmin = w + 1
        xmax = 0

        for x in 1:w
            if labels_im[row, x] == segment_index
                sum_x += x
                n_pixels += 1

                xmin = min(xmin, x)
                xmax = max(xmax, x)
            end
        end

        # Update width statistics if this row contains the segment
        if xmax > 0
            width = xmax - xmin + 1

            n_rows += 1
            delta = width - mean_width
            mean_width += delta / n_rows
            M2_width += delta * (width - mean_width)
        end
    end

    if n_pixels == 0
        return (
            mean_vertical_axis = NaN,
            mean_width = NaN,
            std_width = NaN,
        )
    end

    # Sample standard deviation across rows
    std_width = n_rows > 1 ? sqrt(M2_width / (n_rows - 1)) : 0.0

    return (
        mean_vertical_axis = sum_x / n_pixels,
        mean_width = mean_width,
        std_width = std_width,
    )
end

