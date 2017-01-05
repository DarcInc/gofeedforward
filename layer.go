/*
BSD 2-Clause License

Copyright (c) 2016, Darc Inc
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

package gofeedforward

import (
	"fmt"
	"math/rand"
)

// Core is a square array of values used as weights or other m x n values.  The
// rule of thumb is that inputs are the columns and rows are the outputs.  So
// a set of weights for 3 inputs and 2 outputs would be a [2][3]float64 two
// dimensional array.
type Core [][]float64

// Layer is a layer in a network and is composed of the weights, the last set of
// inputs presented tot he weights and the last output produced by the weights.
type Layer struct {
	Weights Core
	Inputs  []float64
	Outputs []float64
}

// MakeCore creates a new two dimensional array of values.  if inputs are set to
// 3 and outputs are set to 2, this will produce a [2][3]float64 array.
func MakeCore(inputs, outputs int) Core {
	core := make([][]float64, outputs)
	for i := 0; i < outputs; i++ {
		core[i] = make([]float64, inputs)
	}
	return core
}

// Randomize randomizes the set of weights to be values between 0.5 and -0.5.  Note that
// it does not initialize Go's random number generator, which must be done
// in some other setup code.
func (c Core) Randomize() {
	for _, out := range c {
		for i := range out {
			out[i] = rand.Float64() - 0.5
		}
	}
}

// Process takes a set of inputs and produces a set of outputs.  Core is simply doing
// matrix multiplication and does not apply the sigmoid fucntion.
func (c Core) Process(inputs []float64) ([]float64, error) {
	result := make([]float64, len(c))

	for idx := range c {
		if len(c[idx]) != len(inputs) {
			return result, fmt.Errorf("Expected %d inputs but got %d inputs", len(c[idx]), len(inputs))
		}
		result[idx], _ = DotProduct(inputs, c[idx])
	}

	return result, nil
}

// InputSize returns the input size for a set of weights.  This is the raw size of
// the 2 dimensional array.
func (c Core) InputSize() int {
	return len(c[0])
}

// OutputSize returns the output size for a set of weights.  This is the raw size
// of the two dimensional array.
func (c Core) OutputSize() int {
	return len(c)
}

// Add adds two Core arrays of arrays together and returns their result as a separate
// Core.  The cores must be of the same input and output size.
func (c Core) Add(other Core) (Core, error) {
	if c.InputSize() != other.InputSize() || c.OutputSize() != other.OutputSize() {
		return nil, fmt.Errorf("Cannot add a %dx%d to a %dx%d core",
			c.InputSize(), c.OutputSize(), other.InputSize(), other.OutputSize())
	}

	result := MakeCore(c.InputSize(), c.OutputSize())
	for row := range c {
		for col := range c[row] {
			result[row][col] = c[row][col] + other[row][col]
		}
	}
	return result, nil
}

// MakeLayer creates a new layer.  A layer is has an implicit bias input value
// of 1.0, so a layer with 5 inputs and 3 outputs actually needs a weight
// matrix of 3 x 6.
func MakeLayer(inputs, outputs int) Layer {
	return Layer{Weights: MakeCore(inputs+1, outputs)}
}

// Process processes the inputs for a given layer.  It uses the weights to
// produce a weighted sum of the inputs and then uses the Sigmoid transfer
// function to squash the input to a value between 0.0 and 1.0 for each of
// the output 'neurons.'  The input slice is biased by appending a 1.0 to the
// input array.
func (l *Layer) Process(inputs []float64) ([]float64, error) {
	l.Inputs = inputs

	biasedInputs := append(inputs, 1.0)
	outputs, err := l.Weights.Process(biasedInputs)

	if err != nil {
		return nil, err
	}

	for idx := range outputs {
		outputs[idx] = Sigmoid(outputs[idx])
	}
	l.Outputs = make([]float64, len(outputs))
	copy(l.Outputs, outputs)

	return outputs, nil
}

// Randomize randomizes the weights in a layer.  It uses Go's internal
// random number generator and recommends that you initialize the Go random
// number generator prior to using this function.
func (l *Layer) Randomize() {
	l.Weights.Randomize()
}

// UpdateWeights updates the weights in a layer given the Core passed in.  The input size and
// output size of the argument and the layer's weights must match.
func (l *Layer) UpdateWeights(updates Core) error {
	var err error
	l.Weights, err = l.Weights.Add(updates)
	return err
}
