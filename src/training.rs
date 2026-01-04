use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    prelude::Module,
};
use crate::{
    config::ModelConfig,
    data::{MalariaBatcher, MalariaDataset},
    malaria_cnn::MalariaCNN,
};

pub struct MalariaTrainer<B: AutodiffBackend> {
    config: ModelConfig,
    device: B::Device,
}

impl<B: AutodiffBackend> MalariaTrainer<B> {
    pub fn new(config: ModelConfig, device: B::Device) -> Self {
        Self { config, device }
    }

    pub fn run(&self) -> anyhow::Result<()> {
        println!("🚀 Démarrage de l'entraînement sur GPU (WGPU)");
        println!("⚙️  Device: {:?}", self.device);
        
        let model: MalariaCNN<B> = MalariaCNN::new(
            &self.device,
            self.config.image_channels,
            self.config.conv1_filters,
            self.config.conv2_filters,
            self.config.conv3_filters,
            self.config.fc1_units,
            self.config.fc2_units,
            self.config.num_classes,
            self.config.dropout_rate,
        );
        
        println!("✅ Modèle créé");
        println!("📁 Chargement du dataset...");
        
        let full_dataset = MalariaDataset::new(
            "data",
            self.config.image_height,
            self.config.image_width,
            self.config.use_cache,
        )?;
        
        let (train_dataset, valid_dataset) = full_dataset.split(0.8);
        
        println!("📊 Dataset: {} train, {} valid", train_dataset.len(), valid_dataset.len());
        
        // ✅ CORRECTION : Spécifier le Backend dans DataLoaderBuilder
        let batcher_train = MalariaBatcher::<B>::new(
            self.config.image_height,
            self.config.image_width,
        );
        
        let batcher_valid = MalariaBatcher::<B::InnerBackend>::new(
            self.config.image_height,
            self.config.image_width,
        );
        
        // ✅ CORRECTION : DataLoaderBuilder attend le Backend en premier paramètre générique
        let dataloader_train = DataLoaderBuilder::<B, _, _>::new(batcher_train)
            .batch_size(self.config.batch_size)
            .shuffle(42)
            .num_workers(self.config.num_workers)
            .build(train_dataset);
        
        let dataloader_valid = DataLoaderBuilder::<B::InnerBackend, _, _>::new(batcher_valid)
            .batch_size(self.config.batch_size)
            .num_workers(self.config.num_workers)
            .build(valid_dataset);
        
        let optim = AdamConfig::new()
            .with_weight_decay(Some(WeightDecayConfig::new(1e-4)));
        
        println!("⚡ Configuration:");
        println!("   - Époques: {}", self.config.num_epochs);
        println!("   - Batch size: {}", self.config.batch_size);
        println!("   - Learning rate: {}", self.config.learning_rate);
        println!("   - Image size: {}x{}", self.config.image_width, self.config.image_height);
        println!("   - Cache: {}", self.config.use_cache);
        println!("🎯 Démarrage...");
        
        let learner = LearnerBuilder::new("./malaria-model")
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
            .num_epochs(self.config.num_epochs)
            .grads_accumulation(self.config.grad_accum_steps)
            .summary()
            .build(model, optim.init(), self.config.learning_rate);
        
        let model_trained = learner.fit(dataloader_train, dataloader_valid);
        
        println!("💾 Sauvegarde du modèle...");
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model_trained.model.into_record(), "./malaria-model".into())?;
        
        println!("✅ Entraînement terminé!");
        
        Ok(())
    }
}