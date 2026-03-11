use crate::ml::model::PhaseEnsemble;
use ndarray::Array1;
use std::collections::HashMap;

pub struct SurrogateExplainer {
    pub model: PhaseEnsemble,
    pub feature_templates: HashMap<String, String>,
}

impl SurrogateExplainer {
    pub fn new(model: PhaseEnsemble) -> Self {
        let mut feature_templates = HashMap::new();
        let templates = [
            ("material_diff", "Gains a material advantage ({:+.0} cp)"),
            (
                "mobility_us",
                "Increases piece activity and mobility ({:+.0} cp)",
            ),
            (
                "mobility_them",
                "Restricts opponent's piece activity ({:+.0} cp)",
            ),
            (
                "king_ring_pressure_us",
                "Increases attacking pressure near the opponent's king ({:+.0} cp)",
            ),
            (
                "king_ring_pressure_them",
                "Reduces attacking pressure on our own king ({:+.0} cp)",
            ),
            (
                "batteries_us",
                "Forms a powerful battery arrangement ({:+.0} cp)",
            ),
            (
                "outposts_us",
                "Establishes a strong knight outpost ({:+.0} cp)",
            ),
            (
                "bishop_pair_us",
                "Maintains the bishop pair advantage ({:+.0} cp)",
            ),
            (
                "bishop_pair_them",
                "Eliminates the opponent's bishop pair ({:+.0} cp)",
            ),
            ("passed_us", "Creates a dangerous passed pawn ({:+.0} cp)"),
            (
                "passed_them",
                "Successfully blocks or stops an opponent's passed pawn ({:+.0} cp)",
            ),
            (
                "isolated_pawns_us",
                "Avoids creating pawn weaknesses ({:+.0} cp)",
            ),
            (
                "isolated_pawns_them",
                "Forces a pawn weakness (isolated pawn) for the opponent ({:+.0} cp)",
            ),
            (
                "center_control_us",
                "Improves control over the critical central squares ({:+.0} cp)",
            ),
            (
                "center_control_them",
                "Challenges and reduces opponent's central control ({:+.0} cp)",
            ),
            (
                "safe_mobility_us",
                "Safely activates pieces to better squares ({:+.0} cp)",
            ),
            (
                "rook_open_file_us",
                "Positions a rook effectively on an open file ({:+.0} cp)",
            ),
            (
                "backward_pawns_us",
                "Solidifies the pawn structure by fixing a weakness ({:+.0} cp)",
            ),
            (
                "backward_pawns_them",
                "Induces a backward pawn weakness in the opponent's camp ({:+.0} cp)",
            ),
            (
                "pst_us",
                "Optimizes piece placement on the board ({:+.0} cp)",
            ),
            (
                "pst_them",
                "Forces opponent pieces to suboptimal squares ({:+.0} cp)",
            ),
            (
                "pinned_us",
                "Successfully escapes an annoying pin ({:+.0} cp)",
            ),
            (
                "pinned_them",
                "Pins an opponent's piece to create tactical opportunities ({:+.0} cp)",
            ),
            (
                "phase",
                "Strategic move appropriate for the current game phase ({:+.0} cp)",
            ),
        ];

        for (k, v) in templates {
            feature_templates.insert(k.to_string(), v.to_string());
        }

        SurrogateExplainer {
            model,
            feature_templates,
        }
    }

    pub fn explain_move(
        &self,
        features_after: &std::collections::BTreeMap<String, f32>,
        top_k: usize,
        min_cp: f64,
    ) -> Vec<(String, f64, String)> {
        let mut reasons = Vec::new();

        let mut delta_vec = Array1::zeros(self.model.feature_names.len());
        for (i, name) in self.model.feature_names.iter().enumerate() {
            delta_vec[i] = *features_after.get(name).unwrap_or(&0.0) as f64;
        }

        let delta_scaled = if let Some(scaler) = &self.model.scaler {
            let mut s = delta_vec.clone();
            s -= &scaler.mean;
            s /= &scaler.scale;
            s
        } else {
            delta_vec.clone()
        };

        let contributions = self.model.get_contributions(&delta_scaled.view());

        let mut significant = Vec::new();
        for (i, &contrib) in contributions.iter().enumerate() {
            let cp_value: f64 = contrib;
            if cp_value.abs() >= min_cp {
                significant.push((self.model.feature_names[i].clone(), cp_value));
            }
        }

        significant.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        for (name, cp_value) in significant.into_iter().take(top_k) {
            let template = self
                .feature_templates
                .get(&name)
                .cloned()
                .unwrap_or_else(|| format!("{} ({:+.0})", name, cp_value));
            let explanation = template.replace("{:+.0}", &format!("{:+.0}", cp_value));
            reasons.push((name, cp_value, explanation));
        }

        reasons
    }
}
