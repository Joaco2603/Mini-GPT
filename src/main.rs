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

        let mut output:Vec<Vec<f64>> = Vec::new();

        for i in 0..a.len() {
            let mut row: Vec<f64> = Vec::new();
            for j in 0..a[0].len(){
                row.push(a[i][j] + b[i][j]);
            }
            output.push(row);
        }

        Ok(output)
    }

    println!("{:?}",add_matriz(&embeddings, &positions));

    // fn ln(x: f64)-> f64{
    //     if x<= 0.0 {
    //         return f64::NAN;
    //     }

    //     let mut x = x;
    //     let mut k = 0;

    //     while x > 2.0{
    //         x/=2.0;
    //         k+=1;
    //     }
    //     while x < 0.5{
    //         x*=2.0;
    //         k-=1;
    //     }


    // }

    fn positional_angle(
        pos: usize,
        i: usize,
        d_model: usize
    )->f64{
        let mut output: Vec<Vec<f64>> = Vec::new();
        let mut angle:  f64 = 0.0;
        let mut sin:    f64 = 0.0;
        let mut cos:    f64 = 0.0;
        let mut nominator:   usize = 0;
        let mut denominator: f64 = 0.0;

        denominator = power(10000,(2.0 * i as f64 / d_model as f64));
        angle = pos as f64 / denominator;
        angle
    }
}
