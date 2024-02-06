#include "activationlayers.h"

ElementWiseActivationLayer::ElementWiseActivationLayer(myType (*activation)(myType), 
                                                       myType(*derivative)(myType))    
                                                      :activation_function(activation),
                                                       derivative_activation_function(derivative){}    
                                                
Matrix ElementWiseActivationLayer::forward(const Matrix &input){
    previous_activation = input;
    return unaryMatrixOp(activation_function, input);
}

Matrix ElementWiseActivationLayer::backward(const Matrix &derivative_wrt_output){
    return unaryMatrixOp(derivative_activation_function, previous_activation) * derivative_wrt_output;
}

Matrix SoftmaxActivationLayer::forward(const Matrix &input){
    previous_activation = input;
    return matrixVectorOp(softmax, input); 
}

Matrix SoftmaxActivationLayer::backward(const Matrix &derivate_wrt_output){
    //Calculates softmax derivative + backward
    Matrix softmax_previous = matrixVectorOp(softmax, previous_activation);
    return (derivate_wrt_output * softmax_previous * (1.0 - softmax_previous));
}