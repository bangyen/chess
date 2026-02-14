use pyo3::prelude::*;

pub mod eval;
pub mod features;
pub mod pawn_cache;
pub mod search;
pub mod see;
pub mod syzygy;
pub mod zobrist;

/// A Python module implemented in Rust.
#[pymodule]
fn _chess_ai_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(search::find_best_reply, m)?)?;
    m.add_function(wrap_pyfunction!(search::calculate_forcing_swing, m)?)?;
    m.add_function(wrap_pyfunction!(features::extract_features_rust, m)?)?;
    m.add_class::<syzygy::SyzygyTablebase>()?;
    Ok(())
}
