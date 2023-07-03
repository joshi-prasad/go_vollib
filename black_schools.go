package go_vollib

import (
	"fmt"
	"math"

	"github.com/golang/glog"
	"gonum.org/v1/gonum/stat/distuv"
)

const (
	MaxIterations = 100
	Tolerance     = 1e-9
)

type BlackSchools struct {
	IsCallOption         bool
	UnderlyingAssesPrice float64 /* S */
	StrikePrice          float64 /* K */
	AnnulizedVolatility  float64 /* Sigma */
	TimeToExpiryInYear   float64 /* T */
	InterestRate         float64 /* r */

	Price float64
	Delta float64
	Gamma float64
	Vega  float64
	Theta float64
	Rho   float64

	_d1       float64
	_d2       float64
	_a        float64
	_sqrt_t   float64
	_deflater float64
}

func NewBlackSchools(
	is_call_option bool,
	option_price float64,
	underlying_asset_price float64,
	strike_price int64,
	annulized_volatility float64,
	time_to_expiry_in_year float64,
	interest_rate float64) *BlackSchools {
	bs := &BlackSchools{
		IsCallOption:         is_call_option,
		UnderlyingAssesPrice: underlying_asset_price,
		StrikePrice:          float64(strike_price),
		AnnulizedVolatility:  annulized_volatility,
		TimeToExpiryInYear:   time_to_expiry_in_year,
		InterestRate:         interest_rate,
		Price:                option_price,
		Delta:                0,
		Gamma:                0,
		Vega:                 0,
		Theta:                0,
		Rho:                  0,
		_d1:                  0,
		_d2:                  0,
		_a:                   0,
		_sqrt_t:              math.Sqrt(time_to_expiry_in_year),
		_deflater:            0,
	}

	bs._deflater = bs.deflater()
	if annulized_volatility == 0 {
		if bs.Price == 0 {
			glog.Fatal("Both Sigma and Price cannot be zero.")
			return nil
		}
		var err error
		annulized_volatility, err = bs.computeIv()
		if err != nil {
			glog.Error("Failed to converge IV")
			return nil
		}
	}
	if annulized_volatility == 0 {
		glog.Error("Failed to converge IV")
		return nil
	}

	bs.AnnulizedVolatility = annulized_volatility
	bs._a = bs.a(annulized_volatility)
	bs._d1 = bs.d1(annulized_volatility)
	bs._d2 = bs.d2(bs._d1, annulized_volatility)
	if bs.Price == 0 {
		bs.Price = bs.price()
	}
	return bs
}

func (bs *BlackSchools) ComputeGreeks() {
	bs.Delta = bs.delta()
	bs.Gamma = bs.gamma()
	bs.Vega = bs.vega()
	bs.Theta = bs.theta()
	bs.Rho = bs.rho()
}

func (bs *BlackSchools) computeIv() (float64, error) {
	lowVol := 0.0
	highVol := 100.0

	// Implement the Black-Scholes formula
	bsFormula := func(sigma float64) float64 {
		tmpBs := NewBlackSchools(
			bs.IsCallOption,
			0 /* option_price */,
			bs.UnderlyingAssesPrice,
			int64(bs.StrikePrice),
			sigma,
			bs.TimeToExpiryInYear,
			bs.InterestRate)
		return tmpBs.Price
	}

	zeroIntDiffCount := 0
	// Perform a binary search to find the IV
	for i := 0; i < MaxIterations; i++ {
		midVol := (lowVol + highVol) / 2.0
		bsPrice := bsFormula(midVol)
		diff := bsPrice - bs.Price

		absDiff := math.Abs(diff)
		dd, _ := math.Modf(absDiff)
		//fmt.Println(bs.Price, bsPrice, midVol, dd)
		if dd <= 0 {
			zeroIntDiffCount += 1
			if zeroIntDiffCount >= 2 {
				return midVol, nil
			}
		}
		if absDiff <= Tolerance {
			return midVol, nil
		}

		if diff < 0 {
			lowVol = midVol
		} else {
			highVol = midVol
		}
	}

	return 0, fmt.Errorf("IV calculation did not converge")
}

func (bs *BlackSchools) price() float64 {
	if bs.IsCallOption {
		return bs.callPrice(bs._d1, bs._d2)
	}
	return bs.putPrice(bs._d1, bs._d2)
}

func (bs *BlackSchools) callPrice(d1, d2 float64) float64 {
	// price = S * N(d1) - K * exp(-r * T) * N(d2)
	return (bs.UnderlyingAssesPrice * bs.normCdf(d1)) -
		(bs.StrikePrice * bs._deflater * bs.normCdf(d2))
}

func (bs *BlackSchools) putPrice(d1, d2 float64) float64 {
	// price = K * exp(-r * T) * N(-d2) - S * N(-d1)
	return (bs.StrikePrice * bs._deflater * bs.normCdf(-d2)) -
		(bs.UnderlyingAssesPrice * bs.normCdf(-d1))
}

func (bs *BlackSchools) delta() float64 {
	if bs.IsCallOption {
		return bs.normCdf(bs._d1)
	}
	return -bs.normCdf(-bs._d1)
}

func (bs *BlackSchools) gamma() float64 {
	return bs.normPdf(bs._d1) / (bs.UnderlyingAssesPrice * bs._a)
}

func (bs *BlackSchools) vega() float64 {
	return bs.UnderlyingAssesPrice * bs.normPdf(bs._d1) * bs._sqrt_t * 0.01
}

func (bs *BlackSchools) theta() float64 {
	if bs.IsCallOption {
		return bs.callTheta() / 365
	}
	return bs.putTheta() / 365
}

func (bs *BlackSchools) callTheta() float64 {
	// Python formula
	// theta = (-S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T))) -
	//         (r * K * np.exp(-r * T) * norm.cdf(d2, 0, 1))
	return (-bs.UnderlyingAssesPrice * bs.normPdf(bs._d1) *
		bs.AnnulizedVolatility / (2 * bs._sqrt_t)) -
		(bs.InterestRate * bs.StrikePrice * bs._deflater * bs.normCdf(bs._d2))
}

func (bs *BlackSchools) putTheta() float64 {
	// Python formula
	//  theta_calc = (-S * norm.pdf(d1, 0, 1) * sigma / (2 * np.sqrt(T))) +
	//               (r * K * np.exp(-r * T) * norm.cdf(-d2, 0, 1))
	return (-bs.UnderlyingAssesPrice * bs.normPdf(bs._d1) *
		bs.AnnulizedVolatility / (2 * bs._sqrt_t)) +
		(bs.InterestRate * bs.StrikePrice * bs._deflater * bs.normCdf(-bs._d2))
}

func (bs *BlackSchools) rho() float64 {
	if bs.IsCallOption {
		return bs.callRho() * 0.01
	}
	return bs.putRho() * 0.01
}

func (bs *BlackSchools) callRho() float64 {
	// Python
	// rho_calc = K * T * np.exp(-r * T) * norm.cdf(d2, 0, 1)
	return bs.StrikePrice * bs.TimeToExpiryInYear * bs._deflater * bs.normCdf(bs._d2)
}

func (bs *BlackSchools) putRho() float64 {
	// Python formula
	// rho_calc = -K * T * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
	return -bs.StrikePrice * bs.TimeToExpiryInYear * bs._deflater * bs.normCdf(-bs._d2)
}

// Calculates the value of 'a' used in the Black Scholes formula.
// 'a' is computed by multiplying the volatility of the option by the square
// root of the number of days to expiry.
// This value, denoted as 'a' in the Black-Scholes formula, represents the
// standard deviation of the asset's returns over the period.
// This term is a measure of how much the asset price is expected to fluctuate
// over the time to expiry.
func (bs *BlackSchools) a(volatility float64) float64 {
	// Volatility is sigma
	return volatility * bs._sqrt_t
}

// Calculates the value of 'd1' in the Black-Scholes formula.
func (bs *BlackSchools) d1(volatility float64) float64 {
	// The function first calculates the natural logarithm of the ratio of the
	// asset price to the strike price. This term represents the logarithmic
	// return of the asset.
	// The function then calculates the sum of the interest rate and half of the
	// square of the volatility, multiplied by the number of days to expiry.
	// This term represents the risk premium associated with the option.
	// Finally, the function divides the sum by the volatility multiplied by the
	// square root of the time to expiry.

	// The d1 value in the Black-Scholes formula represents the expected
	// percentage increase in the underlying asset price, assuming that the
	// option expires in the money.

	// The d1 value is used in the Black-Scholes formula to calculate the price
	// of a call option. The higher the d1 value, the more expensive the call
	// option will be. This is because a higher d1 value implies that there is a
	// higher probability that the option will expire in the money.

	// Formula
	// d1 = (ln(S / K) + (r + σ² / 2) * T) / σ * √T
	return (math.Log(bs.UnderlyingAssesPrice/bs.StrikePrice) +
		(bs.InterestRate+math.Pow(volatility, 2)/2)*bs.TimeToExpiryInYear) /
		bs._a
}

// CalculateD2Value calculates the value of 'd2' in the Black-Scholes formula.
func (bs *BlackSchools) d2(d1 float64, volatility float64) float64 {
	// 'd2' value is
	// Expected percentage increase in the underlying asset price, assuming that
	// the option expires out of the money
	//
	// d2 = d1 - sigma * np.sqrt(T)
	return d1 - bs._a
}

// Calculate the cumulative distribution function (CDF) of the standard normal
// distribution at a given value. It returns the probability that a random
// variable from a standard normal distribution is less than or equal to the
// specified value.
func (bs *BlackSchools) normCdf(x float64) float64 {
	return distuv.Normal{Mu: 0, Sigma: 1}.CDF(x)
}

// NormPDF calculates the probability density function (PDF) of a standard
// normal distribution at the given value x.
// It uses the `distuv.UnitNormal` distribution from the
// `gonum.org/v1/gonum/stat/distuv` package to compute the PDF.
// The function returns the computed PDF value.
func (self *BlackSchools) normPdf(x float64) float64 {
	// This function is used to calculate the probability density of the
	// underlying asset's returns. It is commonly used to estimate the
	// likelihood of different asset price scenarios and to calculate option
	// Greeks like vega or rho, which represent the sensitivity of the option's
	// price to changes in volatility or interest rates, respectively.
	normalDist := distuv.UnitNormal
	return normalDist.Prob(x)
}

func (bs *BlackSchools) deflater() float64 {
	// deflater is a factor that is used to discount the price of an option to
	// the present value. This is because the option will not be exercised until
	// it expires, and the value of the option will decrease over time due to
	// the risk-free interest rate.

	// The deflater is a significant factor in options pricing. The higher the
	// interest rate, the lower the value of the deflater, and the lower the
	// price of the option. This is because a higher interest rate means that
	// the option will have less value in the future, and therefore is worth
	// less today.

	// The deflater is also affected by the time to expiry of the option. The
	// longer the time to expiry, the lower the value of the deflater, and the
	// lower the price of the option. This is because an option with a longer
	// time to expiry has more time to appreciate in value, and therefore is
	// worth more today.

	// The following table shows the deflater for a call option with a strike
	// price of $100, a risk-free interest rate of 5%, and a time to expiry of
	// 1 year, 2 years, and 3 years:

	// Time to expiry	Deflater
	// 1 year         0.951229
	// 2 years        0.905994
	// 3 years        0.864035

	// As you can see, the deflater decreases as the time to expiry increases.
	// This is because an option with a longer time to expiry has more time to
	// appreciate in value, and therefore is worth more today.
	return math.Exp(-bs.InterestRate * bs.TimeToExpiryInYear)
}
