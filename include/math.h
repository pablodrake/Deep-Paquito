#ifndef MATH_H
#define MATH_H
#include "types.h"

myType product(myType a, myType b);
myType division(myType a, myType b);
myType sum(myType a, myType b);
myType difference(myType a, myType b);
myType sum(const Matrix &input);
myType mean(const Matrix &input);

Matrix unaryMatrixOp(myType (*elemwise_function)(myType), const Matrix &input);
Matrix matrixMatrixOp(myType (*matrixMatrix_elem_wise)(myType, myType), const Matrix &a, const Matrix &b);
Matrix matrixScalarOp(myType (*matrixScalar_elem_wise)(myType, myType), const Matrix &input, myType scalar);
Matrix scalarMatrixOp(myType (*scalarMatrix_elem_wise)(myType, myType), myType scalar, const Matrix &input);
Matrix matrixVectorOp(myType (*matrixVector_elem_wise)(myType, myType), const Matrix &input, const Vector &vector);
Vector vectorVectorOp(myType (*vectorVectorOp_elem_wise)(myType, myType), const Vector &a, const Vector &b);
Vector vectorScalarOp(myType (*vectorScalarOp_elem_wise)(myType, myType), const Vector &input, myType scalar);

Matrix matrixMultiply(const Matrix &a, const Matrix &b);
Matrix matrixTranspose(const Matrix &a);
Matrix operator*(const Matrix &a, const Matrix &b);
Matrix operator/(const Matrix &a, const Matrix &b);
Matrix operator-(const Matrix &a, const Matrix &b);
Matrix operator+(const Matrix &a, const Matrix &b);

Matrix operator*(const Matrix &input, myType scalar);
Matrix operator/(const Matrix &input, myType scalar);
Matrix operator+(const Matrix &input, myType scalar);
Matrix operator-(const Matrix &input, myType scalar);

Matrix operator*(myType scalar, const Matrix &input);
Matrix operator/(myType scalar, const Matrix &input);
Matrix operator+(myType scalar, const Matrix &input);
Matrix operator-(myType scalar, const Matrix &input);

Matrix operator*(const Matrix &input, const Vector &vector);
Matrix operator/(const Matrix &input, const Vector &vector);
Matrix operator+(const Matrix &input, const Vector &vector);
Matrix operator-(const Matrix &input, const Vector &vector);

Vector operator*(const Vector &a, const Vector &b);
Vector operator/(const Vector &a, const Vector &b);
Vector operator-(const Vector &a, const Vector &b);
Vector operator+(const Vector &a, const Vector &b);

Vector operator*(const Vector &input, myType scalar);
Vector operator/(const Vector &input, myType scalar);
Vector operator+(const Vector &input, myType scalar);
Vector operator-(const Vector &input, myType scalar);

myType leakyRelu(myType input);
myType relu(myType input);
myType sigmoid(myType input);
myType meanSquaredError(const Matrix &input, const Matrix &expected_values);

myType leakyReluDerivative(myType input);
myType reluDerivative(myType input);
myType sigmoidDerivative(myType input);
Matrix meanSquaredErrorDerivative(const Matrix &input, const Matrix &expected_values);

void coutMatrix(const Matrix &input);
void coutVector(const Vector &input);

#endif //MATH_H