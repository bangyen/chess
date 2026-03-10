use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2, Axis};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub scale: Array1<f64>,
    pub var: Array1<f64>,
    pub n_samples_seen: usize,
}

impl StandardScaler {
    pub fn new(n_features: usize) -> Self {
        StandardScaler {
            mean: Array1::zeros(n_features),
            scale: Array1::ones(n_features),
            var: Array1::zeros(n_features),
            n_samples_seen: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>) {
        let n_samples = x.nrows();
        self.mean = x.mean_axis(Axis(0)).unwrap();
        self.var = x.var_axis(Axis(0), 0.0);
        self.scale = self.var.mapv(|v| if v > 0.0 { v.sqrt() } else { 1.0 });
        self.n_samples_seen = n_samples;
    }

    pub fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut x_scaled = x.clone();
        for mut row in x_scaled.rows_mut() {
            row -= &self.mean;
            row /= &self.scale;
        }
        x_scaled
    }

    pub fn inverse_transform(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut x_inv = x.clone();
        for mut row in x_inv.rows_mut() {
            row *= &self.scale;
            row += &self.mean;
        }
        x_inv
    }
}
