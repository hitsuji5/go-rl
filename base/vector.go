package base

import (
	"math"
)

type Vector []float64

func (a Vector) Add(b Vector) {
	for i := range a {
		a[i] += b[i]
	}
}

func (a Vector) Sub(b Vector) {
	for i := range a {
		a[i] -= b[i]
	}
}

func (a Vector) Mul(b Vector) {
	for i := range a {
		a[i] *= b[i]
	}
}

func (a Vector) Sum() float64 {
	z := 0.0
	for _, x := range a {
		z += x
	}
	return z
}

func (a Vector) Normalize() {
	z := a.Sum()
	for i := range a {
		a[i] /= z
	}
}

func (a Vector) Norm() float64 {
	norm := 0.0
	for _, x := range a {
		norm += x * x
	}	
	return math.Sqrt(norm)
}

func (a Vector) Fill(x float64) {
	for i := range a {
		a[i] = x
	}
}

func (a Vector) Dot(b Vector) (dot float64) {
	for i := range a {
		dot += a[i] * b[i]
	}
	return
}