#ifndef LAYER_H
#define LAYER_H
#include "types.h"
#include "math.h"

class Layer {
    protected:
        const myType learning_rate = 0.01;
        Matrix previous_activation;
    public:
        static const int batch_size = 8;
        virtual Matrix forward(const Matrix &output);
        virtual Matrix backward(const Matrix &derivative_wrt_output);
};

#endif //LAYER_H