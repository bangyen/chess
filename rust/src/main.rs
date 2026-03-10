use clap::{Parser, Subcommand};
use anyhow::Result;
use std::io::{self, Write};
use _chess_ai_rust::engine::ExplainableEngine;
use _chess_ai_rust::features::extract_features;
use shakmaty::{Chess, Position};

#[derive(Parser)]
#[command(name = "chess-ai")]
#[command(about = "Explainable Chess Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Play an interactive chess game
    Play {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short, long, default_value_t = 12)]
        depth: u32,
    },
    /// Run a feature explainability audit
    Audit {
        #[arg(short, long, default_value = "stockfish")]
        stockfish_path: String,
        #[arg(short, long)]
        fen: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Play { stockfish_path, depth } => {
            let mut engine = ExplainableEngine::new(&stockfish_path)?;
            println!("Welcome to the Explainable Chess Engine (Rust Edition)!");
            
            loop {
                print!("Your move (UCI): ");
                io::stdout().flush()?;
                
                let mut input = String::new();
                io::stdin().read_line(&mut input)?;
                let input = input.trim();
                
                if input == "quit" || input == "exit" {
                    break;
                }

                if let Err(e) = engine.make_move(input) {
                    println!("Error: {}", e);
                    continue;
                }

                println!("Stockfish is thinking...");
                let best_move = engine.get_best_move(depth)?;
                println!("Stockfish plays: {}", best_move);
                engine.make_move(&best_move)?;
            }
        }
        Commands::Audit { stockfish_path, fen } => {
            let fen_str = fen.unwrap_or_else(|| "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string());
            let mut engine = ExplainableEngine::new(&stockfish_path)?;
            
            let pos: Chess = fen_str.parse::<shakmaty::fen::Fen>()?
                .into_position(shakmaty::CastlingMode::Standard)?;
            
            println!("Auditing FEN: {}", fen_str);
            let feats = extract_features(&pos);
            
            println!("\nExtracted Features:");
            for (name, val) in feats {
                println!("  {:30}: {:>8.3}", name, val);
            }
            
            let best_move = engine.get_best_move(12)?;
            println!("\nEngine Recommendation: {}", best_move);
        }
    }

    Ok(())
}
