fn main() {
    let matrix = vec![vec![0.2, 0.5, -0.1], vec![0.8, -0.2, 0.4]];

    fn print_shape(matrix: &Vec<Vec<f32>>) {
        // tu código
        println!("Rows {}", matrix.len());
        println!("Cols {}", matrix[0].len());
    }

    print_shape(&matrix);

    fn get_elements(matrix: &Vec<Vec<f32>>, rows: usize, cols: usize) -> f32 {
        matrix[rows][cols]
    }

    get_elements(&matrix, 1, 1);

    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];

    fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, &'static str> {
        if a.len() != b.len() {
            return Err("Invalid operations the arrays are of different sizes");
        }

        let mut acc: f64 = 0.0;

        for i in 0..a.len() {
            acc += a[i] * b[i];
        }

        Ok(acc)
    }

    println!("{:?}", dot_product(&a, &b));

    let matrix = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

    let vector = vec![2.0, 1.0, 3.0];

    fn matrix_vector_mul(
        matrix: &Vec<Vec<f64>>,
        vector: &Vec<f64>,
    ) -> Result<Vec<f64>, &'static str> {
        let mut output: Vec<f64> = Vec::new();

        for i in 0..matrix.len() {
            match dot_product(&matrix[i], vector) {
                Ok(value) => {
                    output.push(value);
                }
                Err(value) => {
                    return Err(value);
                }
            }
        }

        Ok(output)
    }

    println!("{:?}", matrix_vector_mul(&matrix, &vector));

    let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

    fn get_column(matrix: &Vec<Vec<f64>>, col: usize) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for i in 0..matrix.len() {
            output.push(matrix[i][col]);
        }

        output
    }

    println!("{:?}", get_column(&b, 0));
    // [7.0, 9.0, 11.0]

    let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

    let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

    fn matrix_matrix_mul(
        a: &Vec<Vec<f64>>,
        b: &Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, &'static str> {
        let mut output: Vec<Vec<f64>> = Vec::new();

        if a[0].len() != b.len() {
            return Err("Invalid operations the arrays are of different sizes");
        }

        // FOR cada fila de A
        //     crear una fila vacía
        for i in 0..a.len() {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..b[0].len() {
                let column = get_column(b, j);
                match dot_product(&a[i], &column) {
                    Ok(value) => {
                        row.push(value);
                    }
                    Err(error) => {
                        return Err(error);
                    }
                }
            }

            output.push(row);
        }

        Ok(output)
    }

    println!("{:?}", matrix_matrix_mul(&a, &b));

    let embeddings = vec![
        vec![0.2, 0.5, -0.1], // gato
        vec![0.8, -0.2, 0.4], // come
        vec![0.1, 0.7, 0.3],  // pez
                              // vec![0.9, 0.1, -0.5], // perro
    ];

    let token_ids: Vec<usize> = vec![0, 1, 2];

    fn embedding_lookup(
        embeddings: &Vec<Vec<f64>>,
        token_ids: &Vec<usize>,
    ) -> Result<Vec<Vec<f64>>, &'static str> {
        let mut output: Vec<Vec<f64>> = Vec::new();

        for i in 0..token_ids.len() {
            if token_ids[i] >= embeddings.len() {
                return Err("Bound overflow in memory");
            }
            output.push(embeddings[token_ids[i]].clone());
        }

        Ok(output)
    }

    println!("{:?}", embedding_lookup(&embeddings, &token_ids));

    let positions = vec![
        vec![0.01, 0.02, 0.03],
        vec![0.04, 0.05, 0.06],
        vec![0.07, 0.08, 0.09],
    ];

    fn add_matriz(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, &'static str> {
        if a.len() != b.len() {
            return Err("Different number of rows");
        }

        if a[0].len() != b[0].len() {
            return Err("Different number of columns");
        }

        let mut output: Vec<Vec<f64>> = Vec::new();

        for i in 0..a.len() {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..a[0].len() {
                row.push(a[i][j] + b[i][j]);
            }
            output.push(row);
        }

        Ok(output)
    }

    println!("{:?}", add_matriz(&embeddings, &positions));

    // ln(x) sin usar la librería estándar: ln(x) = ln(2^k * x') = k·ln(2) + ln(x')
    fn ln(x: f64) -> f64 {
        // El logaritmo natural solo está definido para x > 0
        if x <= 0.0 {
            return f64::NAN;
        }

        let mut x = x;
        let mut k = 0; // exponente: x original = 2^k * x (x queda en [0.5, 2])

        // Reducir x dividiendo por 2 mientras sea mayor que 2
        while x > 2.0 {
            x /= 2.0;
            k += 1;
        }
        // Escalar x multiplicando por 2 mientras sea menor que 0.5
        while x < 0.5 {
            x *= 2.0;
            k -= 1;
        }

        // Serie de Taylor para atanh: ln(x) = 2·(y + y³/3 + y⁵/5 + ...), y = (x-1)/(x+1)
        let y = (x - 1.0) / (x + 1.0);
        let y2 = y * y;
        let mut term = y; // término actual de la serie (y, y³, y⁵, ...)
        let mut sum = 0.0;
        let mut n = 1.0; // denominador impar: 1, 3, 5, ...

        for _ in 0..50 {
            sum += term / n;
            term *= y2; // avanzar al siguiente término impar
            n += 2.0;
        }

        // Reconstruir ln(x) sumando la parte de la escala por potencias de 2
        (2.0 * sum) + (k as f64 * 0.6931471805599453) // 0.693... ≈ ln(2)
    }

    fn abs(val: f64) -> f64 {
        if val < 0.0 {
            -val
        } else {
            val
        }
    }

    fn exp(x: f64) -> f64 {
        // 1. Edge case handling according to the IEEE 754 standard
        if x.is_nan() {
            return f64::NAN;
        }
        if x > 709.782712893384 {
            return f64::INFINITY;
        } // Prevents f64 overflow
        if x < -708.3964185322641 {
            return 0.0;
        } // Prevents underflow (flushes to 0.0)

        // Precalculated log2(e) constant
        const LOG2_E: f64 = 1.4426950408889634;

        // 2. Argument reduction to base 2: x * log2(e) = k + r
        let z = x * LOG2_E;
        let k = z.round(); // Integer part (exponent power of 2)
        let r_z = z - k; // Base-2 remainder (-0.5 <= r_z <= 0.5)

        // 3. Convert remainder back to base e: r = r_z * ln(2)
        const LN_2: f64 = 0.6931471805599453;
        let r = r_z * LN_2;

        // 4. Optimized Taylor series for e^r
        // Since |r| <= ~0.346, 13 terms guarantee maximum double-precision accuracy
        let mut sum = 1.0;
        let mut term = 1.0;
        for i in 1..=13 {
            term *= r / (i as f64);
            sum += term;
        }

        // 5. Final reconstruction: sum * 2^k
        // f64::from_bits manipulates the IEEE 754 exponent bits directly in hardware
        let scale = f64::from_bits(((k as i64 + 1023) as u64) << 52);

        sum * scale
    }

    fn int_floor(x: f64) -> f64 {
        let int_part = x as i64 as f64;
        if x < 0.0 && x != int_part {
            int_part - 1.0;
        }
        int_part
    }

    fn power(base: f64, exponent: f64) -> f64 {
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

            if (is_integer) {
                let result = exp(exponent * ln(-base));
                let exp_int = exponent as i64;
                return if exp_int % 2 == 0 { result } else { -result };
            } else {
                return f64::NAN;
            }
        }

        exp(exponent * ln(base))
    }

    fn positional_angle(pos: usize, i: usize, d_model: usize) -> f64 {
        let mut angle: f64 = 0.0;
        let mut denominator: f64 = 0.0;

        denominator = power(10000.0, (2.0 * i as f64 / d_model as f64));
        angle = pos as f64 / denominator;
        angle
    }

    println!("{}", positional_angle(5, 0, 8));
    println!("{}", positional_angle(5, 1, 8));
    println!("{}", positional_angle(5, 2, 8));
    println!("{}", positional_angle(5, 3, 8));

    const PI: f64 = std::f64::consts::PI;
    const TWO_PI: f64 = std::f64::consts::TAU;
    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    #[inline]
    pub fn sin(angle: f64) -> f64 {
        let mut x = angle % TWO_PI;
        if x > PI {
            x -= TWO_PI;
        } else {
            x += TWO_PI;
        }
        let x2 = x * x;
        x * (1.0 - x2 * (1.0 / 6.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 5040.0 - x2 / 362880.0))))
    }

    #[inline]
    pub fn cos(angle: f64) -> f64 {
        sin(angle + (PI / 2.0))
    }

    fn positional_encoding(pos: usize, d_model: usize) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();

        for i in 0..(d_model / 2) {
            let angle = positional_angle(pos, i, d_model);

            output.push(sin(angle));
            output.push(cos(angle));
        }

        output
    }

    fn positional_encoding_matrix(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
        let mut output: Vec<Vec<f64>> = Vec::new();

        for pos in 0..seq_len {
            output.push(positional_encoding(pos, d_model));
        }

        output
    }

    let token_embeddings = embedding_lookup(&embeddings, &token_ids).unwrap();

    let positions = positional_encoding_matrix(token_ids.len(), embeddings[0].len());

    // falta una línea

    let x = match add_matriz(&token_embeddings, &positions) {
        Ok(value) => value,
        Err(err) => {
            eprintln!("{}", err);
            return;
        }
    };

    println!("{:?}", x);

    fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let rows = matrix.len();
        let cols = matrix[0].len();

        let mut output: Vec<Vec<f64>> = Vec::new();

        for i in 0..cols {
            let mut row = Vec::new();

            for j in 0..rows {
                row.push(matrix[j][i]);
            }
            output.push(row);
        }
        output
    }

    let q = vec![
        vec![1.0, 0.0, 1.0, 0.0], // Q token 0
        vec![0.0, 1.0, 1.0, 0.0], // Q token 1
        vec![1.0, 1.0, 0.0, 1.0], // Q token 2
    ];

    let k = vec![
        vec![1.0, 0.0, 1.0, 0.0], // K token 0
        vec![0.0, 1.0, 0.0, 1.0], // K token 1
        vec![1.0, 1.0, 0.0, 0.0], // K token 2
    ];

    let k_t = transpose(&k);

    let scores = matrix_matrix_mul(&q, &k_t).unwrap();

    println!("Kᵀ: {:?}", k_t);
    println!("QKᵀ: {:?}", scores);

    fn scale_matrix(matrix: &Vec<Vec<f64>>, scalar: f64) -> Vec<Vec<f64>> {
        let mut output = Vec::new();

        for i in 0..matrix.len() {
            let mut row = Vec::new();

            for j in 0..matrix[0].len() {
                row.push(matrix[i][j] / scalar);
            }
            output.push(row);
        }

        output
    }

    let d_k = q[0].len();
    let scale = (d_k as f64).sqrt();
    let scaled_scores = scale_matrix(&scores, scale);

    const EULER: f64 = std::f64::consts::E;

    fn softmax(vector: &Vec<f64>) -> Vec<f64> {
        let mut denominator: f64 = 0.0;
        let mut output: Vec<f64> = Vec::new();
        for i in 0..vector.len() {
            denominator += power(EULER, vector[i]);
        }

        for i in 0..vector.len() {
            let numerator = power(EULER, vector[i]);
            output.push(numerator / denominator);
        }

        output
    }

    softmax(&vec![1.0, 0.0, 0.5]);

    for i in 0..scaled_scores.len() {
        softmax(&scaled_scores[i]);
    }

    let mut attention_weights: Vec<Vec<f64>> = Vec::new();

    for i in 0..scaled_scores.len() {
        let row = softmax(&scaled_scores[i]);
        attention_weights.push(row);
    }

    let v = vec![
        vec![1.0, 0.0, 2.0, 0.0], // V token 0
        vec![0.0, 2.0, 0.0, 1.0], // V token 1
        vec![1.0, 1.0, 0.0, 2.0], // V token 2
    ];

    matrix_matrix_mul(&attention_weights, &v);

    let w_q = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.4, 0.5, 0.6],
        vec![0.7, 0.8, 0.9],
    ];

    let w_k = vec![
        vec![0.2, -0.1, 0.4],
        vec![0.5, 0.3, -0.2],
        vec![-0.3, 0.6, 0.1],
    ];

    let w_v = vec![
        vec![0.4, 0.1, -0.2],
        vec![-0.1, 0.5, 0.3],
        vec![0.2, -0.4, 0.6],
    ];

    let q = matrix_matrix_mul(&x, &w_q).unwrap();
    let k = matrix_matrix_mul(&x, &w_k).unwrap();
    let v = matrix_matrix_mul(&x, &w_v).unwrap();

    // 1. Kᵀ
    let k_t = transpose(&k);

    // 2. QKᵀ
    let scores = matrix_matrix_mul(&q, &k_t).unwrap();

    // 3. / √d_k
    let d_k = q[0].len();
    let scale = (d_k as f64).sqrt();

    let scaled_scores = scale_matrix(&scores, scale);

    // 4. Softmax fila por fila
    let mut attention_weights: Vec<Vec<f64>> = Vec::new();

    for i in 0..scaled_scores.len() {
        let row = softmax(&scaled_scores[i]);
        attention_weights.push(row);
    }

    // 5. × V
    let attention_output = matrix_matrix_mul(&attention_weights, &v).unwrap();

    println!("Attention: {:?}", attention_output);

    // seq_len = 3
    // d_model = 4
    // heads = 2

    let x = vec![
        vec![0.2, 0.5, -0.1, 0.4],
        vec![0.8, -0.2, 0.4, 0.1],
        vec![0.1, 0.7, 0.3, -0.5],
    ];

    let w_q1 = vec![
        vec![0.1, 0.1],
        vec![0.3, 0.5],
        vec![0.3, 0.6],
        vec![0.9, -0.2],
    ];

    let w_k1 = vec![
        vec![0.1, 0.2],
        vec![0.4, 0.5],
        vec![0.7, 0.8],
        vec![0.3, -0.2],
    ];

    let w_v1 = vec![
        vec![0.1, -0.2],
        vec![0.6, 0.5],
        vec![0.5, 0.8],
        vec![0.9, -0.2],
    ];

    let w_q2 = vec![
        vec![-0.2, 0.3],
        vec![0.5, -0.1],
        vec![0.2, 0.4],
        vec![-0.4, 0.7],
    ];

    let w_k2 = vec![
        vec![0.3, -0.1],
        vec![0.2, 0.6],
        vec![-0.5, 0.3],
        vec![0.8, 0.1],
    ];

    let w_v2 = vec![
        vec![-0.3, 0.4],
        vec![0.4, -0.3],
        vec![0.7, 0.2],
        vec![-0.1, 0.5],
    ];

    // d_model × d_model → proyecta la concatenación de heads de vuelta a d_model
    let w_o = vec![
        vec![0.2, -0.1, 0.3, 0.4],
        vec![0.5, 0.2, -0.3, 0.1],
        vec![-0.2, 0.6, 0.4, -0.1],
        vec![0.1, 0.3, 0.2, 0.5],
    ];

    fn concat_heads(head1: &Vec<Vec<f64>>, head2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut output = Vec::with_capacity(head1.len());

        for i in 0..head1.len() {
            // Acceso seguro a la fila de head1
            let row1 = match head1.get(i) {
                Some(r) => r.as_slice(),
                None => &[],
            };

            // Acceso seguro a la fila de head2
            let row2 = match head2.get(i) {
                Some(r) => r.as_slice(),
                None => &[],
            };

            let mut row = Vec::with_capacity(row1.len() + row2.len());
            row.extend_from_slice(row1);
            row.extend_from_slice(row2);

            output.push(row);
        }

        output
    }

    fn attention_head(
        x: &Vec<Vec<f64>>,
        w_q: &Vec<Vec<f64>>,
        w_k: &Vec<Vec<f64>>,
        w_v: &Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, &'static str> {
        let q = matrix_matrix_mul(x, w_q)?;
        let k = matrix_matrix_mul(x, w_k)?;
        let v = matrix_matrix_mul(x, w_v)?;

        let k_t = transpose(&k);
        let scores = matrix_matrix_mul(&q, &k_t)?;

        let d_head = q[0].len();
        let scale = (d_head as f64).sqrt();
        let scaled_scores = scale_matrix(&scores, scale);

        let mut attention_weights = Vec::new();
        for i in 0..scaled_scores.len() {
            attention_weights.push(softmax(&scaled_scores[i]));
        }

        matrix_matrix_mul(&attention_weights, &v)
    }

    let head1 = attention_head(&x, &w_q1, &w_k1, &w_v1).unwrap();
    let head2 = attention_head(&x, &w_q2, &w_k2, &w_v2).unwrap();
    let concatenated = concat_heads(&head1, &head2);

    let multihead_output = matrix_matrix_mul(&concatenated, &w_o).unwrap();

    let residual1 = add_matriz(&x, &multihead_output).unwrap();

    fn mean(vector: &Vec<f64>) -> Result<f64, &'static str> {
        if vector.is_empty() {
            return Err("Invalid operation: cannot compute mean of an empty vector");
        }

        let mut output: f64 = 0.0;

        for i in 0..vector.len() {
            output += vector[i];
        }

        Ok(output / vector.len() as f64)
    }

    let x = vec![2.0, 4.0, 6.0, 8.0];

    println!("{}", mean(&x).unwrap());

    fn variance(vector: &Vec<f64>) -> Result<f64, &'static str> {
        if vector.is_empty() {
            return Err("Invalid operation: cannot compute variance of an empty vector");
        }

        // Population variance: (1/n) * sum((xi - mean)^2)
        let m = mean(vector)?;

        let mut sum_squared_diff = 0.0;

        // Iterate each value, subtract the mean, and accumulate the squared difference
        for i in 0..vector.len() {
            let diff = vector[i] - m;
            sum_squared_diff += power(diff, 2.0);
        }

        Ok(sum_squared_diff / vector.len() as f64)
    }

    fn sqrt(x: f64) -> Result<f64, &'static str> {
        if x.is_nan() {
            return Err("Invalid operation: cannot compute square root of NaN");
        }
        if x < 0.0 {
            return Err("Invalid operation: cannot compute square root of a negative number");
        }

        // Square root via exponentiation: x^(1/2)
        Ok(power(x, 0.5))
    }

    fn std_dev(vector: &Vec<f64>) -> Result<f64, &'static str> {
        // Population standard deviation is the square root of the population variance
        let var = variance(vector)?;
        sqrt(var)
    }
}
