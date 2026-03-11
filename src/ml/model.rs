use crate::ml::scaler::StandardScaler;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PhaseModel {
    pub coefficients: Array1<f64>,
    pub intercept: f64,
    pub alpha: f64,
    pub l1_ratio: f64,
}

impl PhaseModel {
    pub fn predict(&self, features: &ArrayView1<f64>) -> f64 {
        self.coefficients.dot(features) + self.intercept
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PhaseEnsemble {
    pub feature_names: Vec<String>,
    pub phase_idx: i32,
    pub models: HashMap<String, PhaseModel>,
    pub global_model: Option<PhaseModel>,
    pub scaler: Option<StandardScaler>,
}

impl PhaseEnsemble {
    pub fn new(feature_names: Vec<String>) -> Self {
        let phase_idx = feature_names
            .iter()
            .position(|r| r == "phase")
            .map(|i| i as i32)
            .unwrap_or(-1);
        PhaseEnsemble {
            feature_names,
            phase_idx,
            models: HashMap::new(),
            global_model: None,
            scaler: None,
        }
    }

    pub fn get_phase(&self, features: &ArrayView1<f64>) -> String {
        if self.phase_idx == -1 {
            return "middlegame".to_string();
        }
        let p = features[self.phase_idx as usize];
        if p > 24.0 {
            "opening".to_string()
        } else if p > 12.0 {
            "middlegame".to_string()
        } else {
            "endgame".to_string()
        }
    }

    pub fn predict(&self, features: &ArrayView1<f64>) -> f64 {
        let phase = self.get_phase(features);
        let model = self.models.get(&phase).or(self.global_model.as_ref());

        match model {
            Some(m) => m.predict(features),
            None => 0.0,
        }
    }

    pub fn get_contributions(&self, features: &ArrayView1<f64>) -> Array1<f64> {
        let phase = self.get_phase(features);
        let model = self.models.get(&phase).or(self.global_model.as_ref());

        match model {
            Some(m) => {
                let mut contribs = m.coefficients.clone();
                for (i, val) in contribs.iter_mut().enumerate() {
                    *val *= features[i];
                }
                contribs
            }
            None => Array1::zeros(features.len()),
        }
    }
}
