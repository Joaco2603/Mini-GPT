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

    fn abs(val: f64)->f64{
        if val < 0.0{
            -val
        }else{
            val
        }
    }

    fn exp(x: f64) -> f64 {
        // 1. Edge case handling according to the IEEE 754 standard
        if x.is_nan() { return f64::NAN; }
        if x > 709.782712893384 { return f64::INFINITY; } // Prevents f64 overflow
        if x < -708.3964185322641 { return 0.0; }         // Prevents underflow (flushes to 0.0)
    
        // Precalculated log2(e) constant
        const LOG2_E: f64 = 1.4426950408889634;
    
        // 2. Argument reduction to base 2: x * log2(e) = k + r
        let z = x * LOG2_E;
        let k = z.round();     // Integer part (exponent power of 2)
        let r_z = z - k;       // Base-2 remainder (-0.5 <= r_z <= 0.5)
    
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

    fn positional_angle(pos: usize, i: usize, d_model: usize) -> f64 {
        let mut output: Vec<Vec<f64>> = Vec::new();
        let mut angle: f64 = 0.0;
        let mut sin: f64 = 0.0;
        let mut cos: f64 = 0.0;
        let mut nominator: usize = 0;
        let mut denominator: f64 = 0.0;

        denominator = power(10000, (2.0 * i as f64 / d_model as f64));
        angle = pos as f64 / denominator;
        angle
    }
}
