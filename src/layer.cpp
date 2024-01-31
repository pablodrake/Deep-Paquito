#include "layer.h"
#include <stdexcept>

Matrix Layer::forward(const Matrix &input){
    throw std::runtime_error("Method not implemented, (layer.forward)");
    return Matrix();
}

Matrix Layer::backward(const Matrix &input){
    throw std::runtime_error("Method not implemented, (layer.backward)");
    return Matrix();
}