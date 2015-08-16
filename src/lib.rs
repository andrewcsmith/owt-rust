extern crate nalgebra as na;
use na::{DVec, DMat, Norm, Transpose, Inv, Diag, Iterable};
use std::iter::{FromIterator, Iterator, IntoIterator};

struct OWTCriteria {
    title: &'static str,
    num_pitches: i32,
    repeat_factor: f64,
    ideal_intervals: DVec<f64>,
    interval_weights: DVec<f64>,
    key_weights: DVec<f64>
}

struct OWTResults {
    chisq: f64,
    optimal_tuning: Vec<f64>
}

impl OWTCriteria {
    fn populate_source_matrix(&self) -> DMat<f64> {
        let nrows = self.num_pitches * (self.num_pitches - 1);
        let ncols = self.num_pitches - 1;
        let mut source_matrix = DMat::new_zeros(nrows as usize, ncols as usize);

        for row_index in 0..nrows {
            let neg_col = (row_index % self.num_pitches) - 1;
            let pos_col = ((row_index % self.num_pitches) + 
                           (row_index / self.num_pitches)) % self.num_pitches;

            if neg_col >= 0 && neg_col < (self.num_pitches - 1) {
                source_matrix[(row_index as usize, neg_col as usize)] = -1.0;
            }
            if pos_col >= 0 && pos_col < (self.num_pitches - 1) {
                source_matrix[(row_index as usize, pos_col as usize)] = 1.0;
            }
        }
        source_matrix
    }

    fn populate_ideal_interval_vector(&self) -> DVec<f64> {
        let vec_size = self.num_pitches * (self.num_pitches - 1);
        DVec::<f64>::from_iter((0..vec_size).map(|i| {
            let base = i / self.num_pitches;
            let incr = i % self.num_pitches;
            let ed = base + incr;
            if ed < (self.num_pitches - 1) {
                self.ideal_intervals[base as usize]
            } else {
                self.ideal_intervals[base as usize] - self.repeat_factor
            }
        }))
    }

    fn populate_weights_vector(&self) -> DVec<f64> {
        let vec_size = self.num_pitches * (self.num_pitches - 1);
        DVec::<f64>::from_iter((0..vec_size).map(|i| {
            let base = i / self.num_pitches;
            let incr = i % self.num_pitches;
            self.interval_weights[base as usize] * self.key_weights[incr as usize]
        }))
    }

    fn optimize_temperament(&self) -> DVec<f64> {
        let source_matrix = self.populate_source_matrix();
        let ideal_intervals_vector = self.populate_ideal_interval_vector();
        let weights_vector = DMat::from_diag(&self.populate_weights_vector());
        match (&source_matrix.transpose() * &weights_vector * &source_matrix).inv() {
            Some(x) => { x * &source_matrix.transpose() * &weights_vector * ideal_intervals_vector },
            None => { panic!("What!!") }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OWTCriteria;
    use na::{DVec, DMat, Iterable};

    fn get_criteria() -> OWTCriteria {
        OWTCriteria {
            title: "Testing",
            num_pitches: 3,
            repeat_factor: 1200.0,
            ideal_intervals: DVec::from_slice(2, &vec![0.0, 702.0]),
            interval_weights: DVec::from_slice(2, &vec![1.0e-6, 1.0]),
            key_weights: DVec::from_slice(3, &vec![1.0, 1.0e-4, 1.0])
        }
    }

    #[test]
    fn test_populate_source_matrix() {
        let criteria = get_criteria();
        let exp = DMat::from_row_vec(6, 2, &vec![
                                     1.0, 0.0,
                                    -1.0, 1.0, 
                                     0.0,-1.0,
                                     0.0, 1.0, 
                                    -1.0, 0.0,
                                     1.0,-1.0]);
        let res = criteria.populate_source_matrix();
        assert_eq!(exp, res);
    }

    #[test]
    fn test_populate_ideal_interval_vector() {
        let criteria = get_criteria();
        let exp = DVec::from_slice(6, &vec![0.0, 0.0, -1200.0, 702.0, -498.0, -498.0]);
        let res = criteria.populate_ideal_interval_vector();
        assert_eq!(exp, res);
    }

    #[test]
    fn test_populate_weights_vector() {
        let criteria = get_criteria();
        let exp = DVec::from_slice(6, &vec![1.0e-6, 1.0e-10, 1.0e-6, 1.0, 1.0e-4, 1.0]);
        let res = criteria.populate_weights_vector();
        assert_eq!(exp, res);
    }

    #[test]
    fn test_optimize_temperament() {
        let criteria = get_criteria();
        let exp = DVec::from_slice(2, &vec![204.059, 702.030]);
        let res = criteria.optimize_temperament();
        for (e, r) in exp.iter().zip(res.iter()) {
            assert!((e - r).abs() < 0.01);
        }
    }
}
