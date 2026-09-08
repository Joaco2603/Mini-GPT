const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = std::f64::consts::TAU;
pub const EULER: f64 = std::f64::consts::E;

/// ln(x) = ln(2^k * x') = k·ln(2) + ln(x'), without using `f64::ln`.
pub fn ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }

    let mut x = x;
    let mut k = 0;

    while x > 2.0 {
        x /= 2.0;
        k += 1;
    }
    while x < 0.5 {
        x *= 2.0;
        k -= 1;
    }

    let y = (x - 1.0) / (x + 1.0);
    let y2 = y * y;
    let mut term = y;
    let mut sum = 0.0;
    let mut n = 1.0;

    for _ in 0..50 {
        sum += term / n;
        term *= y2;
        n += 2.0;
    }

    (2.0 * sum) + (k as f64 * 0.6931471805599453)
}

pub fn abs(val: f64) -> f64 {
    if val < 0.0 {
        -val
    } else {
        val
    }
}

pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 709.782712893384 {
        return f64::INFINITY;
    }
    if x < -708.3964185322641 {
        return 0.0;
    }

    const LOG2_E: f64 = 1.4426950408889634;
    let z = x * LOG2_E;
    let k = z.round();
    let r_z = z - k;

    const LN_2: f64 = 0.6931471805599453;
    let r = r_z * LN_2;

    let mut sum = 1.0;
    let mut term = 1.0;
    for i in 1..=13 {
        term *= r / (i as f64);
        sum += term;
    }

    let scale = f64::from_bits(((k as i64 + 1023) as u64) << 52);
    sum * scale
}

fn int_floor(x: f64) -> f64 {
    let int_part = x as i64 as f64;
    if x < 0.0 && x != int_part {
        int_part - 1.0
    } else {
        int_part
    }
}

pub fn power(base: f64, exponent: f64) -> f64 {
    if base == 0.0 {
        if exponent == 0.0 {
            return 1.0;
        }
        if exponent < 0.0 {
            return f64::INFINITY;
        }
        return 0.0;
    }

    if exponent < 0.0 {
        return 1.0 / power(base, -exponent);
    }

    if base < 0.0 {
        let is_integer = abs(exponent - int_floor(exponent)) < 1e-12;

        if is_integer {
            let result = exp(exponent * ln(-base));
            let exp_int = exponent as i64;
            return if exp_int % 2 == 0 { result } else { -result };
        } else {
            return f64::NAN;
        }
    }

    exp(exponent * ln(base))
}

#[inline]
pub fn sin(angle: f64) -> f64 {
    let mut x = angle % TWO_PI;
    if x > PI {
        x -= TWO_PI;
    } else if x < -PI {
        x += TWO_PI;
    }
    let x2 = x * x;
    x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0 - x2 / 362880.0))))
}

#[inline]
pub fn cos(angle: f64) -> f64 {
    sin(angle + (PI / 2.0))
}

pub fn sqrt(x: f64) -> Result<f64, &'static str> {
    if x.is_nan() {
        return Err("Invalid operation: cannot compute square root of NaN");
    }
    if x < 0.0 {
        return Err("Invalid operation: cannot compute square root of a negative number");
    }
    Ok(power(x, 0.5))
}

pub fn softmax(vector: &[f64]) -> Vec<f64> {
    let mut denominator = 0.0;
    for &value in vector {
        denominator += power(EULER, value);
    }

    let mut output = Vec::new();
    for &value in vector {
        output.push(power(EULER, value) / denominator);
    }
    output
}
