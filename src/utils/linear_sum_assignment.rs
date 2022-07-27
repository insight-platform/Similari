#![cfg(test)]
mod tests {
    use pathfinding::prelude::{kuhn_munkres, Matrix};

    #[test]
    fn lap_hungrian() {
        let weights: Matrix<i64> = Matrix::from_rows(vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]
                .into_iter()
                .map(|x| (x * 1000_000.0) as i64)
                .collect::<Vec<_>>(),
            vec![0.3, 0.8, 0.3, 0.1, 0.0, 0.5, 0.0]
                .into_iter()
                .map(|x| (x * 1000_000.0) as i64)
                .collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.5]
                .into_iter()
                .map(|x| (x * 1000_000.0) as i64)
                .collect::<Vec<_>>(),
            vec![0.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.5]
                .into_iter()
                .map(|x| (x * 1000_000.0) as i64)
                .collect::<Vec<_>>(),
        ])
        .unwrap();

        let (_, assignments) = kuhn_munkres(&weights);
        assert_eq!(assignments, vec![4, 1, 3, 2]);
    }
}
