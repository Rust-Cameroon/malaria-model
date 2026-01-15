use std::env;
use std::{net::SocketAddr, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use axum::{
    extract::DefaultBodyLimit,
    extract::{Multipart, State},
    http::{Method, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use image::{imageops::FilterType, ImageReader};
use serde::Serialize;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, error, info};

// Burn (CPU backend) for native inference
use burn::{
    prelude::Module,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::Tensor,
};
use burn_ndarray::{NdArray, NdArrayDevice};

// Local, inference-only model definition (no training deps)
#[path = "./malaria_cnn_infer.rs"]
mod malaria_cnn;
use malaria_cnn::MalariaCNN;

#[derive(Clone)]
struct AppConfig {
    image_height: usize,
    image_width: usize,
    num_species_classes: usize,
    num_stage_classes: usize,
    stage_loss_lambda: f32,
}

#[derive(Clone)]
struct BurnState {
    cfg: AppConfig,
    model_path: PathBuf,
}

#[derive(Serialize)]
struct PredictResponse {
    infected: bool,
    predicted_species: String,
    species_probabilities: Vec<f32>,
    stage_probabilities: [f32; 4],
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging (RUST_LOG controls level, default to info)
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .compact()
        .init();

    // Inference config must match training
    let cfg = AppConfig {
        image_height: 128,
        image_width: 128,
        num_species_classes: 5,
        num_stage_classes: 4,
        stage_loss_lambda: 0.25,
    };

    // Allow overriding model path via env var MODEL_PATH; default to Burn checkpoint
    let model_path_str =
        env::var("MODEL_PATH").unwrap_or_else(|_| "./malaria-model.bin".to_string());
    let model_path = PathBuf::from(&model_path_str);

    // Proactive existence check to provide a clearer error message
    if !model_path.exists() {
        let cwd = std::env::current_dir().ok();
        let hint = "Expected a Burn checkpoint (.bin). Ensure the file exists or set MODEL_PATH to the checkpoint path.";
        let cwd_msg = cwd
            .map(|p| format!(" Current dir: {}.", p.display()))
            .unwrap_or_default();
        anyhow::bail!(
            "Model checkpoint not found at {}. {}{}",
            model_path.display(),
            hint,
            cwd_msg
        );
    }
    info!(path = %model_path.display(), "Loading Burn checkpoint path configured");
    // Store only config and path in state to keep it Send + Sync
    let state = BurnState { cfg, model_path };

    // CORS: allow dev UIs (Vite default 5173, others) — relax to Any for development simplicity
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::POST, Method::OPTIONS, Method::GET])
        .allow_headers(Any);

    let router = Router::new()
        .route("/health", get(health))
        .route("/predict", post(predict))
        .with_state(state)
        // Multipart uploads can fail to parse if the request body is too large.
        // Set a safe upper limit here (20MB) to avoid sporadic failures on larger images.
        .layer(DefaultBodyLimit::max(20 * 1024 * 1024))
        .layer(cors);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr: SocketAddr = SocketAddr::from(([0, 0, 0, 0], port));
    info!(%addr, "API listening");
    let listener = TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, router)
        .await
        .context("Server error")?;

    Ok(())
}

async fn health() -> impl IntoResponse {
    info!("/health called");
    "ok"
}

async fn predict(State(state): State<BurnState>, mut multipart: Multipart) -> impl IntoResponse {
    let t_total = Instant::now();
    let req_id = uuid::Uuid::new_v4();
    info!(%req_id, "Predict request started");
    // Pull first part named 'image' (also accept 'file' for compatibility)
    let mut image_bytes: Option<Vec<u8>> = None;

    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let is_image_field = matches!(field.name(), Some("image") | Some("file"));
                if !is_image_field {
                    continue;
                }

                match field.bytes().await {
                    Ok(b) => {
                        debug!(%req_id, size = b.len(), "Received image bytes");
                        image_bytes = Some(b.to_vec());
                        break;
                    }
                    Err(e) => {
                        error!(%req_id, error = %e, "Reading multipart field failed");
                        return (
                            StatusCode::BAD_REQUEST,
                            format!(
                                "Invalid image upload: failed to read uploaded file bytes ({})",
                                e
                            ),
                        )
                            .into_response();
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                // This is the common failure when the request is not valid multipart
                // or when the body was truncated (e.g. too large for the configured limit).
                error!(%req_id, error = %e, "Error parsing multipart/form-data request");
                return (
                    StatusCode::BAD_REQUEST,
                    format!(
                        "Invalid image upload: Error parsing `multipart/form-data` request ({})",
                        e
                    ),
                )
                    .into_response();
            }
        }
    }

    let image_bytes = match image_bytes {
        Some(b) => b,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "Invalid image upload: missing form field `image` (or `file`)".to_string(),
            )
                .into_response();
        }
    };

    // Decode and preprocess to CHW f32 [0,1]
    let t_pre = Instant::now();
    let chw = match preprocess_bytes(&image_bytes, state.cfg.image_height, state.cfg.image_width) {
        Ok(v) => v,
        Err(e) => {
            error!(%req_id, error = %e, "Preprocess failed");
            return (StatusCode::BAD_REQUEST, format!("Preprocess failed: {}", e)).into_response();
        }
    };
    debug!(%req_id, ms = t_pre.elapsed().as_millis() as u64, "Preprocessing done");

    // Prepare device and model per request (simple and thread-safe)
    let device = NdArrayDevice::default();

    // Build Burn tensor [1, 3, H, W]
    let input_1d: Tensor<NdArray, 1> = Tensor::<NdArray, 1>::from_floats(chw.as_slice(), &device);
    let input: Tensor<NdArray, 4> =
        input_1d.reshape([1, 3, state.cfg.image_height, state.cfg.image_width]);

    // Instantiate and load model weights
    let t_load = Instant::now();
    let mut model: MalariaCNN<NdArray> = MalariaCNN::new(
        &device,
        3,
        16,
        32,
        64,
        128,
        64,
        state.cfg.num_species_classes,
        state.cfg.num_stage_classes,
        state.cfg.stage_loss_lambda,
        0.3,
    );
    let record = match BinFileRecorder::<FullPrecisionSettings>::new()
        .load(state.model_path.clone(), &device)
    {
        Ok(r) => r,
        Err(e) => {
            error!(%req_id, error = ?e, path = %state.model_path.display(), "Failed to load checkpoint");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to load checkpoint: {:?}", e),
            )
                .into_response();
        }
    };
    model = model.load_record(record);
    debug!(%req_id, ms = t_load.elapsed().as_millis() as u64, "Model loaded");

    let t_inf = Instant::now();
    let (species_logits, stage_logits) = model.forward(input);

    let species_logits_vec: Vec<f32> = match species_logits.into_data().to_vec::<f32>() {
        Ok(v) => v,
        Err(e) => {
            error!(%req_id, error = ?e, "Failed to read species logits");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read species logits: {:?}", e),
            )
                .into_response();
        }
    };

    let stage_logits_vec: Vec<f32> = match stage_logits.into_data().to_vec::<f32>() {
        Ok(v) => v,
        Err(e) => {
            error!(%req_id, error = ?e, "Failed to read stage logits");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read stage logits: {:?}", e),
            )
                .into_response();
        }
    };
    debug!(%req_id, ms = t_inf.elapsed().as_millis() as u64, "Inference done");

    // Species softmax
    let species_probs = softmax(&species_logits_vec);
    if species_probs.len() < state.cfg.num_species_classes {
        error!(%req_id, len = species_probs.len(), expected = state.cfg.num_species_classes, "Invalid species output length");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Invalid species output length",
        )
            .into_response();
    }

    let class_idx = argmax(&species_probs);
    let species_labels = ["Falciparum", "Malariae", "Ovale", "Vivax", "Uninfected"];
    let predicted_species = species_labels
        .get(class_idx)
        .unwrap_or(&"Unknown")
        .to_string();
    let infected = predicted_species != "Uninfected";

    // Stage sigmoid
    let stage_probs_vec = sigmoid_vec(&stage_logits_vec);
    let stage_probabilities = if stage_probs_vec.len() >= 4 {
        [
            stage_probs_vec[0],
            stage_probs_vec[1],
            stage_probs_vec[2],
            stage_probs_vec[3],
        ]
    } else {
        [0.0, 0.0, 0.0, 0.0]
    };

    info!(
        %req_id,
        infected,
        predicted_species,
        total_ms = t_total.elapsed().as_millis() as u64,
        "Prediction ready"
    );

    Json(PredictResponse {
        infected,
        predicted_species,
        species_probabilities: species_probs,
        stage_probabilities,
    })
    .into_response()
}

fn preprocess_bytes(bytes: &[u8], target_height: usize, target_width: usize) -> Result<Vec<f32>> {
    let img = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("Unsupported image format")?
        .decode()
        .context("Failed to decode image")?
        .resize_exact(
            target_width as u32,
            target_height as u32,
            FilterType::Triangle,
        )
        .to_rgb8();

    let raw = img.into_raw();
    let frame = target_height * target_width;
    let mut chw = vec![0.0f32; frame * 3];
    for (i, pix) in raw.chunks_exact(3).enumerate() {
        chw[i] = pix[0] as f32 / 255.0;
        chw[i + frame] = pix[1] as f32 / 255.0;
        chw[i + frame * 2] = pix[2] as f32 / 255.0;
    }
    Ok(chw)
}

fn softmax(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return vec![];
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (*x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

fn sigmoid_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
}
