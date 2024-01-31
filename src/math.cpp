#include "math.h"
#include "types.h"
#include <stdexcept>
#include <iostream>
#include <omp.h>

const myType leaky_slope = 0.01;

Matrix matrixMultiply(const Matrix &a, const Matrix &b){
    if(a[0].size() != b.size()){
        throw std::runtime_error("First matrix row dimension must be equal to the second matrix colunm dimension");
    }
    Matrix output(a.size(), Vector(b[0].size(), 0));

    #pragma omp parallel for collapse(2)
    for(int row = 0; row < a.size(); row++){
        for(int col = 0; col < b[0].size(); col++){

            myType sum = 0;
            for(int k = 0; k < b.size(); k++)
            {
                sum += a[row][k] * b[k][col];
            }
            output[row][col] = sum;
        }
    }
    return output;
}

Matrix matrixTranspose(const Matrix &a){
    Matrix transposed(a[0].size(), Vector(a.size(), 0));

    #pragma omp parallel for collapse(2)
    for(int row = 0; row < a.size(); row++)
    {
        for(int col = 0; col < a[0].size(); col++)
        {
            transposed[col][row] = a[row][col];
        }
    }
    return transposed;
}

myType sum(const Matrix &input){
    myType sum = 0.0;

    for(int row = 0; row < input.size(); row++){
        for(int col = 0; col < input[0].size(); col++){
            sum += input[row][col];
        }
    }
    return sum;
}

myType mean(const Matrix &input){
    return sum(input) / (input.size() * input[0].size());
}

Matrix operator*(const Matrix &a, const Matrix &b){
    Matrix output = matrixMatrixOp(product, a, b);
    return output;
}

Matrix operator/(const Matrix &a, const Matrix &b){
    Matrix output = matrixMatrixOp(division, a, b);
    return output;
}

Matrix operator-(const Matrix &a, const Matrix &b){
    Matrix output = matrixMatrixOp(difference, a, b);
    return output;
}

Matrix operator+(const Matrix &a, const Matrix &b){
    Matrix output = matrixMatrixOp(sum, a, b);
    return output;
}

Matrix operator*(const Matrix &input, myType scalar){
    Matrix output = matrixScalarOp(product, input, scalar);
    return output;
}

Matrix operator/(const Matrix &input, myType scalar){
    Matrix output = matrixScalarOp(division, input, scalar);
    return output;
}

Matrix operator+(const Matrix &input, myType scalar){
    Matrix output = matrixScalarOp(sum, input, scalar);
    return output;
}

Matrix operator-(const Matrix &input, myType scalar){
    Matrix output = matrixScalarOp(difference, input, scalar);
    return output;
}

Matrix operator*(myType scalar, const Matrix &input){
    Matrix output = scalarMatrixOp(product, scalar, input);
    return output;
}

Matrix operator/(myType scalar, const Matrix &input){
    Matrix output = scalarMatrixOp(division, scalar, input);
    return output;
}

Matrix operator+(myType scalar, const Matrix &input){
    Matrix output = scalarMatrixOp(sum, scalar, input);
    return output;
}

Matrix operator-(myType scalar, const Matrix &input){
    Matrix output = scalarMatrixOp(difference, scalar, input);
    return output;
}

Matrix operator*(const Matrix &input, const Vector &vector){
    Matrix output = matrixVectorOp(product, input, vector);
    return output;
}

Matrix operator/(const Matrix &input, const Vector &vector){
    Matrix output = matrixVectorOp(division, input, vector);
    return output;
}

Matrix operator+(const Matrix &input, const Vector &vector){
    Matrix output = matrixVectorOp(sum, input, vector);
    return output;
}

Matrix operator-(const Matrix &input, const Vector &vector){
    Matrix output = matrixVectorOp(difference, input, vector);
    return output;
}

Vector operator*(const Vector &a, const Vector &b){
    Vector output = vectorVectorOp(product, a, b);
    return output;   
}

Vector operator/(const Vector &a, const Vector &b){
    Vector output = vectorVectorOp(division, a, b);
    return output;   
}

Vector operator+(const Vector &a, const Vector &b){
    Vector output = vectorVectorOp(sum, a, b);
    return output;   
}

Vector operator-(const Vector &a, const Vector &b){
    Vector output = vectorVectorOp(difference, a, b);
    return output;   
}

Vector operator*(const Vector & input, myType scalar){
    Vector output = vectorScalarOp(product, input, scalar);
    return output;
}

Vector operator/(const Vector & input, myType scalar){
    Vector output = vectorScalarOp(division, input, scalar);
    return output;
}

Vector operator+(const Vector & input, myType scalar){
    Vector output = vectorScalarOp(sum, input, scalar);
    return output;
}

Vector operator-(const Vector & input, myType scalar){
    Vector output = vectorScalarOp(difference, input, scalar);
    return output;
}

myType leakyRelu(myType input){
    return input > 0 ? input : input*leaky_slope;
}

myType relu(myType input){
    return input > 0 ? input : 0;
}

myType sigmoid(myType input){
    return (1.0/(1.0+exp(-input)));
}

myType leakyReluDerivative(myType input){
    return input > 0 ? 1 : leaky_slope;
}

myType reluDerivative(myType input){
    return input > 9 ? 1 : 0;
}

myType sigmoidDerivative(myType input){
    myType sig = sigmoid(input);
    return sig*(1.0 - sig);
}

myType meanSquaredError(const Matrix &input, const Matrix &expected_values){
    Matrix difference = input - expected_values;
    myType mean_value = mean(difference * difference);
    return mean_value;
}

Matrix meanSquaredErrorDerivative(const Matrix &input, const Matrix &expected_values){
    Matrix output = 2 * (input - expected_values) / (input.size() * input[0].size());
    return output;
}

Matrix unaryMatrixOp(myType (*elemwise_function)(myType), const Matrix &input){
    Matrix output(input.size(), Vector(input[0].size(), 0));

    for(int row = 0; row < input.size(); row++){
        for(int col = 0; col < input[0].size(); col++){
            output[row][col] = elemwise_function(input[row][col]);
        }
    }
    return output;
}


Matrix matrixMatrixOp(myType (*matrixMatrix_elem_wise)(myType, myType), const Matrix &a, const Matrix &b){
    if(a.size() != b.size()){
        throw std::runtime_error("Matrices must have the same size");
    }

    Matrix output(a.size() ,Vector(b[0].size(), 0));
    #pragma omp parallel for collapse(2)
    for(int row = 0; row < a.size(); row++){
        for(int col = 0; col < b[0].size(); col++){
            output[row][col] = matrixMatrix_elem_wise(a[row][col], b[row][col]);
        }
    }
    return output;
}


Matrix matrixScalarOp(myType (*matrixscalar_elem_wise)(myType, myType), const Matrix &input, myType scalar){
    Matrix output(input.size(), Vector(input[0].size(), 0));
    #pragma omp parallel for collapse(2)
    for(int row = 0; row < input.size(); row++){
        for(int col = 0; col < input[0].size(); col++){
            output[row][col] = matrixscalar_elem_wise(input[row][col], scalar);
        }
    }
    return output;
}

Matrix scalarMatrixOp(myType (*matrixscalar_elem_wise)(myType, myType), myType scalar, const Matrix &input){
    Matrix output(input.size(), Vector(input[0].size(), 0));
    #pragma omp parallel for collapse(2)
    for(int row = 0; row < input.size(); row++){
        for(int col = 0; col < input[0].size(); col++){
            output[row][col] = matrixscalar_elem_wise(scalar, input[row][col]);
        }
    }
    return output;
}

Matrix matrixVectorOp(myType (*matrixVector_elem_wise)(myType, myType), const Matrix &input, const Vector &vector){
    if(input[0].size() != vector.size()){
        throw std::runtime_error("Matrix dimension must be equal to vector dimension");
    }

    Matrix output(input.size(), Vector(input[0].size(), 0));
    #pragma omp parallel for collapse(2)
    for(int row = 0; row < input.size(); row++){
        for(int col = 0; col < input[0].size(); col++){
            output[row][col] = matrixVector_elem_wise(input[row][col], vector[col]);
        }
    }

    return output;
}

Vector vectorVectorOp(myType (*vectorVectorOp_elem_wise)(myType, myType),const Vector &a, const Vector &b){
    if(a.size() != b.size()){
        throw std::runtime_error("Vectors must have equal dimensions");
    }

    Vector output(a.size(), 0);
    for(int i = 0; i < a.size(); i++){
        output[i] = vectorVectorOp_elem_wise(a[i], b[i]);
    }
    return output;
}

Vector vectorScalarOp(myType (*vectorScalarOp_elem_wise)(myType, myType), const Vector &input, myType scalar){
    Vector output(input.size(), 0);

    for(int i = 0; i < input.size(); i++){
        output[i] = vectorScalarOp_elem_wise(input[i], scalar);
    }
    return output;
}

myType product(myType a, myType b){
    return a*b;
}

myType division(myType a, myType b){
    return a/b;
}

myType sum(myType a, myType b){
    return a+b;
}

myType difference(myType a, myType b){
    return a-b;
}

void coutMatrix(const Matrix &input){
    for(int row = 0; row < input.size(); row++){
        for(int col = 0; col < input[0].size(); col++){
            std::cout << input[row][col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void coutVector(const Vector &input){
    for(int i = 0; i < input.size(); i++){
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}