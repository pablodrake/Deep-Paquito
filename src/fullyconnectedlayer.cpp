#include "fullyconnectedlayer.h"
#include <iostream>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) : bias(output_size, 0){
    weights = Matrix(input_size, Vector(output_size, 0));
    myType stddev = sqrt(2.0 / input_size);
    std::default_random_engine generator(rand());
    std::normal_distribution<myType> distribution(0, stddev);

    for(int i = 0; i < input_size; i++){
        for(int j = 0; j < output_size; j++){
            weights[i][j] = distribution(generator);
        }
    }
}

Matrix FullyConnectedLayer::forward(const Matrix &input){
    previous_activation = input;
    Matrix output = matrixMultiply(input, weights) + bias;
    return output;
}

Matrix FullyConnectedLayer::backward(const Matrix &derivative_wrt_output){
    //Back propagation
    //derivada de la función de coste con respecto a la matriz de pesos
    size_t rows = previous_activation.size();
    Vector derivative_wrt_bias(derivative_wrt_output[0].size(), 0);

    for(int i = 0; i < derivative_wrt_output.size(); i++){
        derivative_wrt_bias = derivative_wrt_bias + derivative_wrt_output[i];
    }

    Matrix derivative_wrt_weights = matrixMultiply(matrixTranspose(previous_activation), derivative_wrt_output);
    Matrix derviative_wrt_input = matrixMultiply(derivative_wrt_output, matrixTranspose(weights));

    //Optimization(gradient descent)
    weights = weights - (derivative_wrt_weights * learning_rate);
    bias = bias - (derivative_wrt_bias * learning_rate);

    return derviative_wrt_input;
}

void FullyConnectedLayer::coutLayer(){
    coutMatrix(weights);
    coutVector(bias);
}