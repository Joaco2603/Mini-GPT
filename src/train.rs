use crate::libs::math::{ln, softmax};

/// Mean cross-entropy over sequence positions.
///
/// `logits`:  [seq_len × vocab_size]
/// `targets`: [seq_len]  target token id at each position
///
/// For each row i: loss_i = -ln(softmax(logits[i])[targets[i]])
/// Return the mean of those seq_len values.
pub fn cross_entropy_loss(logits: &[Vec<f64>], targets: &[usize]) -> Result<f64, &'static str> {
    if logits.is_empty() {
        return Err("logits must be non-empty");
    }
    if targets.is_empty() {
        return Err("targets must be non-empty");
    }
    if logits.len() != targets.len() {
        return Err("logits and targets must have the same length");
    }
    if logits[0].is_empty() {
        return Err("logits rows must be non-empty");
    }

    let vocab_size = logits[0].len();
    let mut total = 0.0;

    for (row, &target) in logits.iter().zip(targets.iter()) {
        if row.is_empty() {
            return Err("logits rows must be non-empty");
        }
        if row.len() != vocab_size {
            return Err("All logits rows must have the same vocab_size");
        }
        if target >= vocab_size {
            return Err("target token id is out of vocab");
        }

        let probs = softmax(row);
        let p = if probs[target] < 1e-12 {
            1e-12
        } else {
            probs[target]
        };
        total += -ln(p);
    }

    Ok(total / logits.len() as f64)
}

/// Length `vocab_size`, `1.0` at `target`, `0.0` elsewhere.
#[allow(unused_variables)]
pub fn one_hot(vocab_size: usize, target: usize) -> Result<Vec<f64>, &'static str> {
    // validate target < vocab_size
    if target >= vocab_size {
        return Err("Target index is out of bounds for the given vocabulary size");
    }

    let mut vec = vec![0.0; vocab_size];
    vec[target] = 1.0;
    Ok(vec)
}

/// dL/dlogits para la entropía cruzada media (mean cross-entropy). Misma dimensión que `logits`: [seq × vocab].
///
/// Para cada fila: (softmax(row) - one_hot(target)) / seq_len
///
/// Dividir por seq_len coincide con `cross_entropy_loss` (promedio, no suma).
#[allow(unused_variables)]
pub fn cross_entropy_grad_logits(
    logits: &[Vec<f64>],
    targets: &[usize],
) -> Result<Vec<Vec<f64>>, &'static str> {
    // 1. VALIDACIONES INICIALES
    // - Validar que 'logits' no esté vacío.
    if logits.is_empty() {
        return Err("Logits is empty");
    }
    if targets.is_empty() {
        return Err("targets must be non-empty");
    }
    // - Validar que 'logits.len() == targets.len()' (la secuencia debe coincidir con las etiquetas).
    if logits.len() != targets.len() {
        return Err("Logits and targets must have the same length");
    }
    if logits[0].is_empty() {
        return Err("logits rows must be non-empty");
    }
    // - Obtener la dimensión del vocabulario (vocab_size) de la primera fila y asegurar
    //   que todas las demás filas tengan la misma longitud.
    let vocab_size = logits[0].len();

    // - Validar que cada elemento en 'targets' sea menor que 'vocab_size' (target < vocab_size).
    for row in logits {
        if row.is_empty() {
            return Err("Logits rows must be non-empty");
        }
        if row.len() != vocab_size {
            return Err("All logits rows must have the same ");
        }
    }

    // 2. OBTENER LA LONGITUD DE LA SECUENCIA (seq_len)
    // Se usará para promediar el gradiente al final.
    // let seq_len = logits.len() as f64;
    let seq_len = logits.len() as f64;

    // 3. MATRIZ DE RESULTADO
    // Crear la matriz contenedora para almacenar los gradientes [seq x vocab].
    let mut grads = Vec::new();

    // 4. BUCLE PARA PROCESAR CADA ELEMENTO DE LA SECUENCIA (Fila por fila)
    // Para i de 0 a logits.len():
    for i in 0..logits.len() {
        //   a. Obtener los logits de la posición actual: logits[i]
        //   b. Obtener el índice objetivo actual: targets[i]
        //
        let target_i = targets[i];

        //   c. Calcular Softmax sobre la fila actual:
        //      probs = softmax(&logits[i])?
        //
        let probs = softmax(&logits[i]);

        //   d. Generar el vector One-Hot para la etiqueta actual:
        //      oh = one_hot(vocab_size, target_i)?
        //
        let oh = one_hot(vocab_size, target_i)?;

        let mut grad_row = vec![0.0; vocab_size];

        //   e. Calcular la diferencia ponderada para cada clase en el vocabulario (j):
        //      grad_row[j] = (probs[j] - oh[j]) / seq_len
        //
        for j in 0..vocab_size {
            grad_row[j] = (probs[j] - oh[j]) / seq_len;
        }

        grads.push(grad_row);
    }
    Ok(grads)
}

/// `weights[i][j] -= lr * grads[i][j]`
#[allow(unused_variables)]
pub fn sgd_update_matrix(
    weights: &mut [Vec<f64>],
    grads: &[Vec<f64>],
    lr: f64,
) -> Result<(), &'static str> {
    // validate same shape, then subtract
    todo!("SGD step")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_logits_is_neg_ln_half() {
        let logits = vec![vec![0.0, 0.0]];
        let loss = cross_entropy_loss(&logits, &[0]).unwrap();
        let expected = -ln(0.5);
        assert!(
            (loss - expected).abs() < 1e-6,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn mean_over_two_positions() {
        let logits = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let loss = cross_entropy_loss(&logits, &[0, 1]).unwrap();
        let expected = -ln(0.5);
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn rejects_length_mismatch() {
        let logits = vec![vec![0.0, 1.0]];
        assert!(cross_entropy_loss(&logits, &[0, 1]).is_err());
    }

    #[test]
    fn rejects_target_out_of_vocab() {
        let logits = vec![vec![0.0, 1.0]];
        assert!(cross_entropy_loss(&logits, &[2]).is_err());
    }
}
