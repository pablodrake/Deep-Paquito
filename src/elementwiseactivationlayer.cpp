#include "elementwiseactivationlayer.h"

ElementWiseActivationLayer::ElementWiseActivationLayer(myType (*activation)(myType), 
                                                       myType(*derivative)(myType))    
                                                      :activation_function(activation),
                                                       derivative_activation_function(derivative){}    
                                                
Matrix ElementWiseActivationLayer::forward(const Matrix &input){
    previous_activation = input;
    Matrix output = unaryMatrixOp(activation_function, input);
    return output;
}

Matrix ElementWiseActivationLayer::backward(const Matrix &derivative_wrt_output){
    Matrix derivative = unaryMatrixOp(derivative_activation_function, previous_activation) * derivative_wrt_output;
    return derivative;
}