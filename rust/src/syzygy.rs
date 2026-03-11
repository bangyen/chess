#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use shakmaty_syzygy::{Tablebase, AmbiguousWdl, MaybeRounded};
#[cfg(feature = "python")]
use shakmaty::{Chess, Fen, CastlingMode};

#[cfg(feature = "python")]
#[pyclass]
pub struct SyzygyTablebase {
    tb: Tablebase<Chess>,
}

#[cfg(feature = "python")]
#[pymethods]
impl SyzygyTablebase {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let mut tb = Tablebase::new();
        tb.add_directory(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(SyzygyTablebase { tb })
    }

    fn probe_wdl(&self, fen: &str) -> PyResult<Option<i32>> {
        let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

        match self.tb.probe_wdl(&pos) {
            Ok(wdl) => match wdl {
                AmbiguousWdl::Win => Ok(Some(2)),
                AmbiguousWdl::Loss => Ok(Some(-2)),
                AmbiguousWdl::Draw => Ok(Some(0)),
                AmbiguousWdl::BlessedLoss => Ok(Some(-1)),
                AmbiguousWdl::CursedWin => Ok(Some(1)),
                AmbiguousWdl::MaybeWin => Ok(Some(1)),
                AmbiguousWdl::MaybeLoss => Ok(Some(-1)),
            },
            Err(_) => Ok(None),
        }
    }

    fn probe_dtz(&self, fen: &str) -> PyResult<Option<i32>> {
        let setup: Fen = fen.parse().map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid FEN"))?;
        let pos: Chess = setup.into_position(CastlingMode::Standard).map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid Position"))?;

        match self.tb.probe_dtz(&pos) {
            Ok(maybe_rounded) => {
                let dtz = match maybe_rounded {
                    MaybeRounded::Precise(d) => d,
                    MaybeRounded::Rounded(d) => d,
                };
                Ok(Some(dtz.0 as i32))
            },
            Err(_) => Ok(None),
        }
    }
}
