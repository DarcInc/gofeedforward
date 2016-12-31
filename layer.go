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
	"math/rand"
	"fmt"
)

type Core [][]float64

type Layer struct {
	Weights Core
	Inputs []float64
	Outputs []float64
}

func MakeCore(inputs, outputs int) Core {
	core := make([][]float64, outputs)
	for i := 0; i < outputs; i++ {
		core[i] = make([]float64, inputs)
	}
	return core
}

func (c Core) Randomize() {
	for _, out := range c {
		for i := range out {
			out[i] = rand.Float64()
		}
	}
}

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

func (c Core) InputSize() int {
	return len(c[0])
}

func (c Core) OutputSize() int {
	return len(c)
}

func MakeLayer(inputs, outputs int ) Layer {
	return Layer{Weights: MakeCore(inputs + 1, outputs)}
}

func (l *Layer) Process(inputs []float64) ([]float64, error) {
	l.Inputs = inputs

	biasedInputs := append(inputs, 1.0)
	outputs, _ := l.Weights.Process(biasedInputs)
	for idx := range outputs {
		outputs[idx] = Sigmoid(outputs[idx])
	}
	l.Outputs = outputs

	return outputs, nil
}

func (l *Layer) Randomize() {
	l.Weights.Randomize()
}