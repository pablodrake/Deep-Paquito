#ifndef ELEMENTWISEACTIVATIONLAYER_H
#define ELEMENTWISEACTIVATIONLAYER_H

#include "layer.h"

class ElementWiseActivationLayer : public Layer{
    private:
        myType (*activation_function)(myType);
        myType (*derivative_activation_function)(myType);

    public:
        ElementWiseActivationLayer(myType (*activation)(myType), myType(*derivative)(myType));
        Matrix forward(const Matrix &input);
        /**
         * Devuelve la derivada con respecto a la entrada
         * Entra la derivada de la funcion de costo con respecto a la salida de esta capa
        */
        Matrix backward(const Matrix &derivative_wrt_output);
};


#endif //ELEMENTWISEACTIVATIONLAYER_H