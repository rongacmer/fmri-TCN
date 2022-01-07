def floyd(matrix):
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i):
            for k in range(num_nodes):
                matrix[i,j] = min(matrix[i,j],max(matrix[i,k],matrix[k,j]))
                matrix[j,i] = matrix[i,j]
    return matrix