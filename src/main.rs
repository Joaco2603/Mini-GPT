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
}
