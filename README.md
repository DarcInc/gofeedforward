# Go Feed Forward
With the emphasis on simple, this is a feed forward libary implemented in Go.

## Theory of operation
A feed forward network is designed to classify inputs using the combination
of weighted sums of those inputs.  The weights for the sums are calculated 
from random initial values using backpropagation.  

backpropagation basically assigns credit to each weight for the amount of 
error that weighted contributed to the final error.  Using a combination
of the derivative of the error function and the magnitude of the weight, the weights
are updated to reduce the error between the inputs and expected outputs
from a set of known samples.

See [Backpropagation on Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
for a fairly detailed discussion of the principles of backpropagation.

## Creating a network layer
A network is composed of layers.  A layer is responsible for taking a set of 
inputs and producing a set of outputs.  

```golang
layer := gofeedforward.MakeLayer(3, 2)
```

This creates a layer with three inputs and 2 outputs.  At first blush this 
would seemd to be 6 weights.  However, because the network inputs need to be
biased, there are actual 8 weights.  When the data is presented to the 
layer an implicit 1 is added to the array of inputs.

## Presenting data to the network layer
A layer is presnted data through a slice of <code>float64</code> values.

```golang
output, err := layer.Process([]float64{0.25, 0.35, 0.75})
if err != nil {
    ...
}
```