package gofeedforward

import "testing"

func TestIrisTraining(t *testing.T) {
	IrisData.Scale(0)
	IrisData.Scale(1)
	IrisData.Scale(2)
	IrisData.Scale(3)

}
