



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

