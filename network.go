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

// Network represents a neural network.  It is composed of its layers and the
// output from the last inputs presented.
type Network struct {
	Layers  []Layer
	Outputs []float64
}

// MakeNetwork returns a neural network with the given size layers.  For example, a 2, 4, 1
// network will have two inputs, 4 hidden layer neurons, and 1 output neurons.
func MakeNetwork(sizes ...int) Network {
	result := Network{}
	for i := 1; i < len(sizes); i++ {
		inputs := sizes[i-1]
		outputs := sizes[i]
		result.Layers = append(result.Layers, MakeLayer(inputs, outputs))
	}
	return result
}

// Randomize updates the weights in the network to random values.
func (n *Network) Randomize() {
	for _, layer := range n.Layers {
		layer.Randomize()
	}
}

// Process takes the given input and produces a set of outputs for the network.
func (n *Network) Process(inputs []float64) ([]float64, error) {
	n.Outputs = nil
	var err error
	temp := inputs
	for idx := range n.Layers {
		temp, err = n.Layers[idx].Process(temp)
		if err != nil {
			return nil, err
		}
	}
	n.Outputs = make([]float64, len(temp))
	copy(n.Outputs, temp)
	return n.Outputs, nil
}

// InputSize returns the network input size
func (n Network) InputSize() int {
	return n.Layers[0].Weights.InputSize() - 1
}

// OutputSize returns the network output size
func (n Network) OutputSize() int {
	return n.Layers[len(n.Layers)-1].Weights.OutputSize()
}
