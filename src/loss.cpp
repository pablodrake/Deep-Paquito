#include "loss.h"
#include "math.h"
#include <stdexcept>

myType Loss::forward(const Matrix &output, const Matrix &expected_output) const{
    throw std::runtime_error("Method is not implemented, (Loss::forward)");
    return 0;
}

Matrix Loss::backward(const Matrix &output, const Matrix &expected_output) const{
    throw std::runtime_error("Method is not implemented, (Loss::backward)");
    return Matrix();
}

myType MSE::forward(const Matrix &output, const Matrix &expected_output) const{
    return meanSquaredError(output, expected_output);
}

Matrix MSE::backward(const Matrix &output, const Matrix &expected_output) const{
    return meanSquaredErrorDerivative(output, expected_output);
}
