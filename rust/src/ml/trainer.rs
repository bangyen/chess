use anyhow::{Result, anyhow};
use shakmaty::{Chess, Position, Role, Square};
use rand::seq::SliceRandom;
use rand::Rng;
use ndarray::{Array1, Array2, Axis};
use linfa::prelude::*;
use linfa_elasticnet::ElasticNet;

use crate::engine::UciEngine;
use crate::features::extract_features;
use crate::ml::model::{PhaseEnsemble, PhaseModel};
use crate::ml::scaler::StandardScaler;

pub fn board_phase_value(pos: &Chess) -> i32 {
    let board = pos.board();
    let mut val = 0;
    for sq in Square::ALL {
        if let Some(piece) = board.piece_at(sq) {
            match piece.role {
                Role::Queen | Role::Rook | Role::Bishop | Role::Knight => val += 1,
                _ => {}
            }
        }
    }
    val
}

pub fn classify_phase(pos: &Chess) -> String {
    let phase = board_phase_value(pos);
    if phase >= 12 {
        "opening".to_string()
    } else if phase >= 6 {
        "middlegame".to_string()
    } else {
        "endgame".to_string()
    }
}

pub fn generate_stratified_positions(n: usize) -> Vec<Chess> {
    let mut rng = rand::thread_rng();
    let targets = [
        ("opening", (n as f32 * 0.25) as usize),
        ("middlegame", (n as f32 * 0.50) as usize),
        ("endgame", (n as f32 * 0.25) as usize),
    ];

    let mut boards = Vec::new();
    for (phase, target) in targets {
        let mut count = 0;
        let mut attempts = 0;
        let max_attempts = target * 50;
        
        while count < target && attempts < max_attempts {
            attempts += 1;
            let mut b = Chess::default();
            let plies = match phase {
                "opening" => rng.gen_range(4..14),
                "middlegame" => rng.gen_range(15..35),
                _ => rng.gen_range(36..60),
            };

            for _ in 0..plies {
                let moves = b.legal_moves();
                if moves.is_empty() || b.is_game_over() { break; }
                
                if phase == "endgame" {
                    let captures: Vec<_> = moves.iter().filter(|m| m.is_capture()).collect();
                    if !captures.is_empty() && rng.gen_bool(0.6) {
                        b.play_unchecked(**captures.choose(&mut rng).unwrap());
                        continue;
                    }
                }
                b.play_unchecked(**moves.iter().collect::<Vec<_>>().choose(&mut rng).unwrap());
            }

            if !b.is_game_over() && classify_phase(&b) == phase {
                boards.push(b);
                count += 1;
            }
        }
    }
    boards.shuffle(&mut rng);
    boards
}

pub fn train_surrogate_model(
    engine_path: &str,
    n_positions: usize,
) -> Result<PhaseEnsemble> {
    let mut engine = UciEngine::new(engine_path)?;
    let boards = generate_stratified_positions(n_positions);
    
    let mut X_raw = Vec::new();
    let mut y_raw = Vec::new();
    let mut feature_names = Vec::new();
    let mut set_feature_names = false;

    for (i, b) in boards.iter().enumerate() {
        if (i + 1) % 10 == 0 {
            println!("Processing position {}/{}...", i + 1, n_positions);
        }
        
        let fen = shakmaty::fen::Fen::from_position(b, shakmaty::EnPassantMode::Always).to_string();
        let base_eval_res = engine.get_evaluation(&fen, 12);
        if base_eval_res.is_err() { continue; }
        let base_eval = base_eval_res.unwrap();
        
        let top_moves_res = engine.get_top_moves(&fen, 12, 3);
        if top_moves_res.is_err() { continue; }
        let top_moves = top_moves_res.unwrap();

        for (mv_uci, _score) in top_moves {
            let mut b_after = b.clone();
            let uci_move: shakmaty::uci::UciMove = match mv_uci.parse() {
                Ok(m) => m,
                Err(_) => continue,
            };

            if let Ok(m) = uci_move.to_move(b) {
                b_after.play_unchecked(m);
                
                let fen_after = shakmaty::fen::Fen::from_position(&b_after, shakmaty::EnPassantMode::Always).to_string();
                let best_reply_uci_res = engine.get_best_move(&fen_after, 12);
                if best_reply_uci_res.is_err() { continue; }
                let best_reply_uci = best_reply_uci_res.unwrap();

                let reply_uci: shakmaty::uci::UciMove = match best_reply_uci.parse() {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                if let Ok(rm) = reply_uci.to_move(&b_after) {
                    b_after.play_unchecked(rm);
                    
                    let fen_final = shakmaty::fen::Fen::from_position(&b_after, shakmaty::EnPassantMode::Always).to_string();
                    let after_eval_res = engine.get_evaluation(&fen_final, 12);
                    if after_eval_res.is_err() { continue; }
                    let after_eval = after_eval_res.unwrap();

                    let delta = after_eval - base_eval;
                    
                    let feats = extract_features(&b_after);
                    if !set_feature_names {
                        feature_names = feats.keys().cloned().collect();
                        set_feature_names = true;
                    }
                    
                    let mut row = Vec::new();
                    for name in &feature_names {
                        row.push(*feats.get(name).unwrap_or(&0.0) as f64);
                    }
                    X_raw.push(row);
                    y_raw.push(delta as f64);
                }
            }
        }
    }

    if X_raw.is_empty() {
        return Err(anyhow!("No training data collected"));
    }

    let n_samples = X_raw.len();
    let n_features = feature_names.len();
    let X_mat = Array2::from_shape_vec((n_samples, n_features), X_raw.into_iter().flatten().collect())?;
    let y_vec = Array1::from_vec(y_raw);

    let mut scaler = StandardScaler::new(n_features);
    scaler.fit(&X_mat);
    let X_scaled = scaler.transform(&X_mat);

    let mut ensemble = PhaseEnsemble::new(feature_names);
    ensemble.scaler = Some(scaler);

    println!("Training global model with {} samples...", n_samples);
    let dataset = Dataset::new(X_scaled.clone(), y_vec.clone());
    let (best_alpha, best_l1) = cross_validate_elastic_net(&dataset)?;
    
    let global_model = ElasticNet::params()
        .penalty(best_alpha)
        .l1_ratio(best_l1)
        .fit(&dataset)
        .map_err(|e| anyhow!("Failed to fit global model: {}", e))?;
    
    ensemble.global_model = Some(PhaseModel {
        coefficients: global_model.hyperplane().clone(),
        intercept: global_model.intercept(),
        alpha: best_alpha,
        l1_ratio: best_l1,
    });

    for phase_name in ["opening", "middlegame", "endgame"] {
        let mut idx = Vec::new();
        for i in 0..n_samples {
            if i < boards.len() && classify_phase(&boards[i]) == phase_name {
                idx.push(i);
            }
        }
        
        if idx.len() > 20 {
            println!("Training model for {} ({} samples)...", phase_name, idx.len());
            let X_phase = X_scaled.select(Axis(0), &idx);
            let y_phase = y_vec.select(Axis(0), &idx);
            let ds_phase = Dataset::new(X_phase, y_phase);
            let (pa, pl1) = cross_validate_elastic_net(&ds_phase)?;
            let m = ElasticNet::params()
                .penalty(pa)
                .l1_ratio(pl1)
                .fit(&ds_phase)
                .map_err(|e| anyhow!("Failed to fit phase model: {}", e))?;
            
            ensemble.models.insert(phase_name.to_string(), PhaseModel {
                coefficients: m.hyperplane().clone(),
                intercept: m.intercept(),
                alpha: pa,
                l1_ratio: pl1,
            });
        }
    }

    Ok(ensemble)
}

fn cross_validate_elastic_net(dataset: &Dataset<f64, f64, ndarray::Ix1>) -> Result<(f64, f64)> {
    let alphas = [0.01, 0.1, 1.0, 10.0];
    let l1_ratios = [0.5, 0.7, 0.9, 1.0];
    
    if dataset.nsamples() < 10 {
        return Ok((0.1, 1.0)); // Default params for very small data
    }

    let mut best_score = f64::MAX;
    let mut best_params = (0.1, 1.0);

    for &a in &alphas {
        for &l1 in &l1_ratios {
            let mut total_err = 0.0;
            let k = 3.min(dataset.nsamples()); // Use smaller k for small datasets
            for i in 0..k {
                let ratio = 1.0 - 1.0 / (k - i) as f32;
                if ratio <= 0.0 || ratio >= 1.0 { continue; }
                let (train, val) = dataset.clone().split_with_ratio(ratio);
                
                if train.nsamples() == 0 || val.nsamples() == 0 { continue; }

                let model = ElasticNet::params()
                    .penalty(a)
                    .l1_ratio(l1)
                    .fit(&train);
                
                if let Ok(m) = model {
                    let preds = m.predict(val.records());
                    for (p, t) in preds.iter().zip(val.targets().iter()) {
                        total_err += (p - t).powi(2);
                    }
                }
            }
            let avg_err = total_err / dataset.nsamples() as f64;
            if avg_err < best_score {
                best_score = avg_err;
                best_params = (a, l1);
            }
        }
    }
    Ok(best_params)
}
