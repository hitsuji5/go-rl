package base


func CosineSimilarity(a []float64, b []float64) float64 {
	return Vector(a).Dot(Vector(b)) / Vector(a).Norm() / Vector(b).Norm()
}

func MeanSquaredError(yTrue, yPred []float64) float64 {
	mse := 0.0
	for i := range yTrue {
		x := yTrue[i] - yPred[i]
		mse += x * x
	}
	mse /= float64(len(yTrue))
	return mse
}