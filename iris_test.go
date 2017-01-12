package gofeedforward

import "testing"

func TestIrisTraining(t *testing.T) {
	IrisData.Scale(0, 1, 2)
	IrisData.Scale(3)

	n := MakeNetwork(4, 6, 3)
	n.Randomize()

	IrisData.Shuffle(10)
	training, tv, err := IrisData.Split(0.6667)
	if err != nil {
		t.Errorf("Failed to split iris training set: %v", err)
	}
	if err != nil {
		t.Errorf("Failed to split test and valiation: %v", err)
	}

	trainer := Trainer{Alpha: 0.1, BatchUpdate: true}
	trainer.AddSimpleStoppingCriteria(10000, 0.1)
	trainer.Train(&n, training)

	classError, err := ClassificationError(n, tv, MakeBestOfClassifier([]string{"virginica", "versicolor", "setosa"}))
	if err != nil {
		t.Errorf("Failed to get classification error: %v", err)
	}

	if classError > 0.10 {
		t.Errorf("Failed to properly train network")
	}
}
