#ifndef LOSS_H
#define LOSS_H
#include "types.h"


class Loss{
private:
public:
    virtual myType forward(const Matrix &output, const Matrix &expected_output) const;
    virtual Matrix backward(const Matrix &output, const Matrix &expected_output) const;
};

class MSE : public Loss{
    public:
    myType forward(const Matrix &output, const Matrix &expected_output) const override;
    Matrix backward(const Matrix &output, const Matrix &expected_output) const override;
};

#endif //LOSS_H

