#include "neuralnetwork.h"
#include <iostream>
NeuralNetwork::NeuralNetwork(const std::vector<std::shared_ptr<Layer>> &layers) : layers(layers){}


Matrix NeuralNetwork::forward(const Matrix &input){
    Matrix previous_output = input;
    for(int i = 0; i < layers.size(); i++){
        previous_output = layers[i]->forward(previous_output);
    }
    return previous_output;
}

Matrix NeuralNetwork::backward(const Matrix &derivative_wrt_out){
    Matrix derivative_wrt_output = derivative_wrt_out;
    for(int i = layers.size() - 1; i >= 0; i--){
        derivative_wrt_output = layers[i]->backward(derivative_wrt_output);
    }
    return derivative_wrt_output;
}