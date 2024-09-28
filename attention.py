# I didn't actually use this code because MLX has a natively defined function for fast multihead attention.
# But I'm proud of it, so I'm keeping it.

import mlx.core as mx
import mlx.nn as nn

# Make QKV matrices all the same size for convenience

def single_vector_attention(input_vector, key_matrix, query_matrix, value_matrix):
    model_dimension = len(query_matrix[0])
    if len(query_matrix[0]) != len(key_matrix[0]):
        print("Model dimensions do not match")

    attention_vector = []
    
    # Top row of query matrix needs to match dimension of model input embeddings
    for i in range(len(model_dimension)):
        query = input_vector @ query_matrix
        for j in range(i):
            key = input_vector @ key_matrix
            attention = mx.transpose(key) / ( mx.sqrt(model_dimension))
            attention = mx.softmax(attention)
            attention_vector.append(attention)

    return attention_vector


def single_head_attention_slow(inverse_square_root_of_dimension, input_vector, key_matrix, query_matrix, value_matrix):

    attention_vector = mx.softmax((query_matrix @ mx.transpose(key_matrix)) * inverse_square_root_of_dimension) @ value_matrix

    return attention_vector

def single_head_attention(inverse_square_root_of_dimension, input_vector, key_matrix, query_matrix, value_matrix):

    attention_vector = mx.matmul(mx.softmax(mx.multiply(mx.matmul(query_matrix, mx.transpose(key_matrix, stream=mx.gpu), stream=mx.gpu), inverse_square_root_of_dimension, stream=mx.gpu)), value_matrix, stream=mx.gpu)

    return attention_vector
