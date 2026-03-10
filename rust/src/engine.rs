use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use shakmaty::{Chess, Move, fen::Fen, Position};
use anyhow::{Result, anyhow};
use std::time::Duration;

pub struct UciEngine {
    process: Child,
}

impl UciEngine {
    pub fn new(path: &str) -> Result<Self> {
        let process = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let mut engine = UciEngine { process };
        engine.send_command("uci")?;
        engine.wait_for_line("uciok", Duration::from_secs(5))?;
        Ok(engine)
    }

    pub fn send_command(&mut self, cmd: &str) -> Result<()> {
        let stdin = self.process.stdin.as_mut().ok_or_else(|| anyhow!("Failed to open stdin"))?;
        writeln!(stdin, "{}", cmd)?;
        stdin.flush()?;
        Ok(())
    }

    pub fn wait_for_line(&mut self, expected: &str, _timeout: Duration) -> Result<String> {
        let stdout = self.process.stdout.as_mut().ok_or_else(|| anyhow!("Failed to open stdout"))?;
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        
        while reader.read_line(&mut line)? > 0 {
            if line.contains(expected) {
                return Ok(line);
            }
            line.clear();
        }
        Err(anyhow!("Timeout/EOF waiting for {}", expected))
    }

    pub fn get_best_move(&mut self, fen: &str, depth: u32) -> Result<String> {
        self.send_command(&format!("position fen {}", fen))?;
        self.send_command(&format!("go depth {}", depth))?;
        let line = self.wait_for_line("bestmove", Duration::from_secs(30))?;
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "bestmove" {
            Ok(parts[1].to_string())
        } else {
            Err(anyhow!("Unexpected response from engine: {}", line))
        }
    }
}

pub struct ExplainableEngine {
    uci: UciEngine,
    pos: Chess,
    history: Vec<Move>,
}

impl ExplainableEngine {
    pub fn new(stockfish_path: &str) -> Result<Self> {
        let uci = UciEngine::new(stockfish_path)?;
        Ok(ExplainableEngine {
            uci,
            pos: Chess::default(),
            history: Vec::new(),
        })
    }

    pub fn make_move(&mut self, move_uci: &str) -> Result<()> {
        let uci_move: shakmaty::uci::UciMove = move_uci.parse()
            .map_err(|e| anyhow!("Invalid move format {}: {:?}", move_uci, e))?;
        
        let m = uci_move.to_move(&self.pos)
            .map_err(|e| anyhow!("Illegal or invalid move {}: {:?}", move_uci, e))?;
        
        self.pos.play_unchecked(m);
        self.history.push(m);
        Ok(())
    }

    pub fn get_best_move(&mut self, depth: u32) -> Result<String> {
        let fen = Fen::from_position(&self.pos, shakmaty::EnPassantMode::Always).to_string();
        self.uci.get_best_move(&fen, depth)
    }
}
