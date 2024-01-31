#include "fullyconnectedlayer.h"
#include <iostream>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) : bias(output_size, 0), t(0){
    weights = Matrix(input_size, Vector(output_size, 0));
    velocity_weights = Matrix(input_size, Vector(output_size, 0));
    moment_weights = Matrix(input_size, Vector(output_size, 0));
    moment_bias = Vector(bias.size(), 0);
    velocity_bias = Vector(bias.size(), 0);
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
    //Previous optimizer

    /**
     *  weights = weights - (derivative_wrt_weights * learning_rate);
     *   bias = bias - (derivative_wrt_bias * learning_rate);
    */
    
    t++;
    moment_weights = beta1 * moment_weights + (1.0 - beta1) * derivative_wrt_weights;
    velocity_weights = beta2 * velocity_weights + (1.0 - beta2) * derivative_wrt_weights * derivative_wrt_weights;

    Matrix moment_weights_hat = moment_weights / (1.0 - pow(beta1, t));
    Matrix velocity_weights_hat = velocity_weights / (1.0 - pow(beta2, t));

    weights = weights - ((learning_rate * moment_weights_hat / (unaryMatrixOp(std::sqrt, velocity_weights_hat) + epsilon)) / batch_size);

    //
    
    moment_bias = moment_bias * beta1 + derivative_wrt_bias * (1.0 - beta1);
    velocity_bias = velocity_bias * beta2 + derivative_wrt_bias * derivative_wrt_bias * (1.0 - beta2);

    Vector moment_bias_hat = moment_bias / (1.0 - pow(beta1, t));
    Vector velocity_bias_hat = velocity_bias / (1.0 - pow(beta2, t));

    bias = bias - ((moment_bias_hat * learning_rate / (unaryVectorOp(std::sqrt, velocity_bias_hat) + epsilon)) / batch_size);

    return derviative_wrt_input;
}

void FullyConnectedLayer::coutLayer(){
    coutMatrix(weights);
    coutVector(bias);
}