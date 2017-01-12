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

import "testing"

func TestMakeNetwork(t *testing.T) {
	net := MakeNetwork(3, 4, 1)

	if len(net.Layers) != 2 {
		t.Errorf("Should have been 2 network layers, instead got: %d", len(net.Layers))
	}

	if net.Layers[0].Weights.InputSize() != 3 && net.Layers[0].Weights.OutputSize() != 4 {
		t.Errorf("Layer sizes are invalid.  Expected %d -> %d but got %d -> %d", 3, 4,
			net.Layers[0].Weights.InputSize(), net.Layers[0].Weights.OutputSize())
	}

	if net.Layers[1].Weights.InputSize() != 4 && net.Layers[1].Weights.OutputSize() != 1 {
		t.Errorf("Layer sizes are invalid.  Expected %d -> %d but got %d -> %d", 4, 1,
			net.Layers[1].Weights.InputSize(), net.Layers[1].Weights.OutputSize())
	}
}

func TestNetwork_Randomize(t *testing.T) {
	net := MakeNetwork(2, 3, 1)
	for _, layer := range net.Layers {
		for _, row := range layer.Weights {
			for _, val := range row {
				if outOfBoundsCheck(0.0, val, 0.001) {
					t.Errorf("Expected 0.0 but got %0.4f", val)
				}
			}
		}
	}

	net.Randomize()

	for _, layer := range net.Layers {
		for _, row := range layer.Weights {
			for _, val := range row {
				if !outOfBoundsCheck(0.0, val, 0.001) {
					t.Errorf("Expected not 0.0 but got %0.4f", val)
				}
			}
		}
	}
}

func TestNetwork_Process(t *testing.T) {
	net := MakeNetwork(2, 3, 1)
	inputs := []float64{1.0, 1.0}

	outputs, _ := net.Process(inputs)
	if len(outputs) != 1 {
		t.Errorf("Expected output size to be 1 but got %d", len(outputs))
	}
}

func TestNetwork_ProcessInvalidInputSize(t *testing.T) {
	net := MakeNetwork(2, 3, 1)
	inputs := []float64{1.0, 1.0, 1.0}

	outputs, err := net.Process(inputs)
	if outputs != nil {
		t.Errorf("Expected outputs to be nil but got %v", outputs)
	}

	if err == nil {
		t.Error("Expected error to be non nil")
	}
}

func TestMakeBestOfClassifier(t *testing.T) {
	classifier := MakeBestOfClassifier([]string{"one", "two", "three"})

	if cl, _ := classifier([]float64{0.1, 0.5, 0.2}); cl[0] != "two" {
		t.Errorf("Expected 'two' but got '%s'", cl)
	}

	if cl, _ := classifier([]float64{0.9, 0.1, 0.8}); cl[0] != "one" {
		t.Errorf("Expected 'one' but got '%s'", cl)
	}

	if cl, _ := classifier([]float64{0.9, 0.1, 0.9}); len(cl) != 1 {
		t.Errorf("There should only be one value but got %d", len(cl))
	}
}

func TestMakeThresholdClassifier(t *testing.T) {
	classifier := MakeThresholdClassifier([]string{"one", "two", "three"}, 0.7)

	if cl, _ := classifier([]float64{0.5, 0.6, 0.1}); len(cl) > 0 {
		t.Errorf("There should be no classes returned but got %v", cl)
	}

	if cl, _ := classifier([]float64{0.71, 0.72, 0.73}); len(cl) != 3 {
		t.Errorf("There should have been three classes returned by got %v", cl)
	}

	cl, _ := classifier([]float64{0.5, 0.71, 0.72})
	if cl[0] != "two" {
		t.Errorf("Expected 'two' but got %s", cl[0])
	}

	if cl[1] != "three" {
		t.Errorf("Expected 'three' but got '%s'", cl[1])
	}
}
