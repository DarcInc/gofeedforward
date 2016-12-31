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

func TestMakeCore(t *testing.T) {
	core := MakeCore(3, 5)

	if len(core) != 5 {
		t.Errorf("Expected core output size to be 5 but was %d", len(core))
	}

	for _, v := range core {
		if len(v) != 3 {
			t.Errorf("Expected core input size was the but got %d", len(v))
		}
	}
}

func TestRandomizeCore(t *testing.T) {
	core := MakeCore(2, 1)

	for _, l := range core {
		for _, v := range l {
			if outOfBoundsCheck(0.0, v, 0.001) {
				t.Errorf("Expected the default core value to be 0 but got %0.4f", v)
			}
		}
	}

	core.Randomize()

	for _, l := range core {
		for _, v := range l {
			if !outOfBoundsCheck(0.0, v, 0.001) {
				t.Errorf("Expected the default core value to be 0 but got %0.4f", v)
			}
		}
	}
}


func TestProcessCore(t *testing.T) {
	core := MakeCore(3, 1)
	core[0][0] = 1.0
	core[0][1] = 1.0
	core[0][2] = 1.0

	i := []float64{1.0, 2.0, 3.0}
	actual, _ := core.Process(i)

	if outOfBoundsCheck(6.0, actual[0], 0.001) {
		t.Errorf("Expected output to be 6.0 but got %0.4f", actual)
	}
}

func TestProcessCoreSizeError(t *testing.T) {
	core := MakeCore(2, 1)
	core[0][0] = 1.0
	core[0][1] = 1.0

	i := []float64{2.0, 3.0, 4.0}
	_, err := core.Process(i)

	if err == nil {
		t.Error("Expected an error but got no error")
	}
}

func TestCore_InputSize(t *testing.T) {
	core := MakeCore(2, 1)
	if core.InputSize() != 2 {
		t.Errorf("Expected 2 inputs but got %d", core.InputSize())
	}
}

func TestCore_OutputSize(t *testing.T) {
	core := MakeCore(2, 1)
	if core.OutputSize() != 1 {
		t.Errorf("Expected 1 output but got %d", core.OutputSize())
	}
}

func TestMakeLayer(t *testing.T) {
	l := MakeLayer(2, 1)

	if l.Weights.InputSize() != 3 {
		t.Errorf("Expected a biased input size of 3 but got %d", l.Weights.InputSize())
	}

	if l.Weights.OutputSize() != 1 {
		t.Errorf("Expected an output size of 1 but got %d", l.Weights.OutputSize())
	}
}

func TestLayer_ProcessSaveInputs(t *testing.T) {
	l := MakeLayer(2, 1)
	inputs := []float64{1.0, 2.0}

	l.Process(inputs)

	if outOfBoundsCheck(1.0, l.Inputs[0], 0.001) {
		t.Errorf("Expected input value to be preserved expected 1.0 but got %0.4f", l.Inputs[0])
	}
}

func TestLayer_Randomize(t *testing.T) {
	l := MakeLayer(2, 1)

	for _, c := range l.Weights {
		for _, v := range c {
			if outOfBoundsCheck(0.0, v, 0.001) {
				t.Errorf("Expected 0 value but got %0.4f", v)
			}
		}
	}

	l.Randomize()

	for _, c := range l.Weights {
		for _, v := range c {
			if !outOfBoundsCheck(0.0, v, 0.001) {
				t.Errorf("Expected non-zero value but got %0.4f", v)
			}
		}
	}
}

func TestLayer_Process(t *testing.T) {
	l := MakeLayer(2, 1)

	outputs, _ := l.Process([]float64{1.0, 2.0})
	if outOfBoundsCheck(0.5, outputs[0], 0.001) {
		t.Errorf("Expected 0.5 but got %d", outputs[0])
	}
}

func TestLayer_ProcessKeepOutputs(t *testing.T) {
	l := MakeLayer(2, 1)

	l.Process([]float64{1.0, 2.0})
	if outOfBoundsCheck(0.5, l.Outputs[0], 0.001) {
		t.Errorf("Expected 0.5 but got %d", l.Outputs[0])
	}
}

