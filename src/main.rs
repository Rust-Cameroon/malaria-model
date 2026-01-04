mod malaria_cnn;
mod training;
mod config;
mod data;

use anyhow::Result;
use burn::backend::{wgpu::{Wgpu, WgpuDevice}, Autodiff};
use crate::training::MalariaTrainer;
use crate::config::ModelConfig;

type Backend = Autodiff<Wgpu<f32, i32>>;

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════╗");
    println!("║  🔥 BURN + WGPU - STABLE & PRODUCTION-READY ║");
    println!("╚════════════════════════════════════════════╝");
    
    // ✅ RÈGLE D'OR : Device créé UNE SEULE FOIS
    let device = WgpuDevice::default();
    
    // ✅ Configuration GPU-SAFE
    let config = ModelConfig {
        image_width: 128,       // Taille safe
        image_height: 128,      // Taille safe
        batch_size: 4,          // Petit au début
        num_epochs: 15,
        use_cache: true,        // Performance
        num_workers: 2,         // Stabilité
        learning_rate: 0.001,
        ..Default::default()
    };
    
    println!("\n📋 Configuration:");
    println!("   • Image: {}x{}", config.image_width, config.image_height);
    println!("   • Batch size: {}", config.batch_size);
    println!("   • Cache: activé");
    println!("   • Device: {:?}\n", device);
    
    // ✅ Device partagé partout
    let trainer = MalariaTrainer::<Backend>::new(config, device);
    trainer.run()
}