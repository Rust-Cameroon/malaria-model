use anyhow::{anyhow, Result};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use csv::Reader;
use image::{imageops::FilterType, ImageReader};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Dataset item containing image path and label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MalariaItem {
    pub image_path: String,
    pub label: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct MpIdbItem {
    pub image_path: String,
    pub infected: u8,
    pub species_id: u8,
    pub stage_labels: [u8; 4],
    pub source_image_id: String,
}

/// Dataset for malaria detection
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MalariaDataset {
    pub images: Vec<PathBuf>,
    pub labels: Vec<u8>,
    pub cache: Option<Arc<HashMap<PathBuf, Vec<f32>>>>,
    pub target_height: usize,
    pub target_width: usize,
    pub use_cache: bool,
}

#[allow(dead_code)]
impl MalariaDataset {
    pub fn new<P: AsRef<Path>>(
        root_dir: P,
        target_height: usize,
        target_width: usize,
        use_cache: bool,
    ) -> Result<Self> {
        let root_dir = root_dir.as_ref();
        println!("📂 Loading dataset from: {}", root_dir.display());

        let parasitized_dir = root_dir.join("Parasitized");
        let uninfected_dir = root_dir.join("Uninfected");

        if !parasitized_dir.exists() {
            return Err(anyhow!(
                "Missing folder: {}/Parasitized/",
                root_dir.display()
            ));
        }
        if !uninfected_dir.exists() {
            return Err(anyhow!(
                "Missing folder: {}/Uninfected/",
                root_dir.display()
            ));
        }

        let mut images = Vec::new();
        let mut labels = Vec::new();

        Self::load_images_from_dir(&parasitized_dir, 1, &mut images, &mut labels)?;
        Self::load_images_from_dir(&uninfected_dir, 0, &mut images, &mut labels)?;

        let mut rng = StdRng::seed_from_u64(42);
        let mut combined: Vec<_> = images.into_iter().zip(labels).collect();
        combined.shuffle(&mut rng);
        let (shuffled_images, shuffled_labels): (Vec<_>, Vec<_>) = combined.into_iter().unzip();

        println!("📊 Dataset loaded: {} total images", shuffled_images.len());
        println!(
            "   - Parasitized: {}",
            shuffled_labels.iter().filter(|&&l| l == 1).count()
        );
        println!(
            "   - Uninfected: {}",
            shuffled_labels.iter().filter(|&&l| l == 0).count()
        );

        let dataset = Self {
            images: shuffled_images,
            labels: shuffled_labels,
            cache: None,
            target_height,
            target_width,
            use_cache,
        };

        let dataset = if use_cache {
            println!("💾 Initializing cache...");
            let cache = dataset.build_cache();
            println!("✅ Cache initialized with {} images.", cache.len());
            Self {
                cache: Some(Arc::new(cache)),
                ..dataset
            }
        } else {
            dataset
        };

        Ok(dataset)
    }

    fn build_cache(&self) -> HashMap<PathBuf, Vec<f32>> {
        let mut cache = HashMap::with_capacity(self.images.len());
        for path in &self.images {
            if let Ok(data) =
                Self::load_and_preprocess_image_raw(path, self.target_height, self.target_width)
            {
                cache.insert(path.clone(), data);
            }
        }
        cache
    }

    fn load_images_from_dir(
        dir: &Path,
        label: u8,
        images: &mut Vec<PathBuf>,
        labels: &mut Vec<u8>,
    ) -> Result<()> {
        let entries = fs::read_dir(dir)
            .map_err(|e| anyhow!("Error reading directory {}: {}", dir.display(), e))?;

        let mut count = 0;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if matches!(
                        ext.as_str(),
                        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
                    ) {
                        images.push(path);
                        labels.push(label);
                        count += 1;
                    }
                }
            }
        }
        println!("   - {} : {} images loaded", dir.display(), count);
        Ok(())
    }

    /// ✅ CPU ONLY - Optimized preprocessing
    pub fn load_and_preprocess_image_raw(
        path: &Path,
        target_height: usize,
        target_width: usize,
    ) -> Result<Vec<f32>> {
        let img = ImageReader::open(path)?.decode()?.resize_exact(
            target_width as u32,
            target_height as u32,
            FilterType::Triangle,
        );

        let rgb_img = img.to_rgb8();
        let raw_pixels = rgb_img.into_raw();
        let frame_size = target_height * target_width;
        let total_size = frame_size * 3;

        let mut chw_data = vec![0.0; total_size];
        for (i, chunk) in raw_pixels.chunks_exact(3).enumerate() {
            chw_data[i] = chunk[0] as f32 / 255.0;
            chw_data[i + frame_size] = chunk[1] as f32 / 255.0;
            chw_data[i + 2 * frame_size] = chunk[2] as f32 / 255.0;
        }

        Ok(chw_data)
    }

    pub fn get_image_data(&self, index: usize) -> Result<Vec<f32>> {
        if index >= self.images.len() {
            return Err(anyhow!("Index {} out of bounds", index));
        }

        let path = &self.images[index];
        if self.use_cache {
            if let Some(cache) = &self.cache {
                if let Some(data) = cache.get(path) {
                    return Ok(data.clone());
                }
            }
            Self::load_and_preprocess_image_raw(path, self.target_height, self.target_width)
        } else {
            Self::load_and_preprocess_image_raw(path, self.target_height, self.target_width)
        }
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn split(&self, ratio: f32) -> (Self, Self) {
        assert!(ratio > 0.0 && ratio < 1.0, "Ratio must be between 0 and 1");
        let split_index = (self.images.len() as f32 * ratio) as usize;

        let train_images = self.images[..split_index].to_vec();
        let train_labels = self.labels[..split_index].to_vec();
        let valid_images = self.images[split_index..].to_vec();
        let valid_labels = self.labels[split_index..].to_vec();

        println!("📈 Dataset split (ratio: {}):", ratio);
        println!("   - Training: {} images", train_images.len());
        println!("   - Validation: {} images", valid_images.len());

        let train_ds = Self {
            images: train_images,
            labels: train_labels,
            cache: self.cache.clone(),
            target_height: self.target_height,
            target_width: self.target_width,
            use_cache: self.use_cache,
        };

        let valid_ds = Self {
            images: valid_images,
            labels: valid_labels,
            cache: self.cache.clone(),
            target_height: self.target_height,
            target_width: self.target_width,
            use_cache: self.use_cache,
        };

        (train_ds, valid_ds)
    }

    pub fn get(&self, index: usize) -> Option<MalariaItem> {
        if index < self.images.len() {
            Some(MalariaItem {
                image_path: self.images[index].to_string_lossy().to_string(),
                label: self.labels[index],
            })
        } else {
            None
        }
    }
}

impl Dataset<MalariaItem> for MalariaDataset {
    fn get(&self, index: usize) -> Option<MalariaItem> {
        self.get(index)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Debug, Clone)]
pub struct MpIdbDataset {
    pub items: Vec<MpIdbItem>,
    pub cache: Option<Arc<HashMap<PathBuf, Vec<f32>>>>,
    pub target_height: usize,
    pub target_width: usize,
    pub use_cache: bool,
}

impl MpIdbDataset {
    pub fn from_manifest<P: AsRef<Path>>(
        manifest_path: P,
        target_height: usize,
        target_width: usize,
        use_cache: bool,
    ) -> Result<Self> {
        let manifest_path = manifest_path.as_ref();

        let mut rdr = Reader::from_path(manifest_path)
            .map_err(|e| anyhow!("Failed to open manifest {}: {}", manifest_path.display(), e))?;

        let mut items: Vec<MpIdbItem> = Vec::new();
        for rec in rdr.records() {
            let rec = rec.map_err(|e| anyhow!("Failed to read manifest record: {}", e))?;
            let crop_path = rec
                .get(0)
                .ok_or_else(|| anyhow!("manifest missing crop_path"))?
                .to_string();
            let infected: u8 = rec
                .get(1)
                .ok_or_else(|| anyhow!("manifest missing infected"))?
                .parse()
                .map_err(|_| anyhow!("Invalid infected"))?;

            let species = rec
                .get(2)
                .ok_or_else(|| anyhow!("manifest missing species"))?;

            let species_id = if infected == 0 {
                4
            } else {
                species_to_id(species)
                    .ok_or_else(|| anyhow!("Unknown species '{}' in manifest", species))?
            };

            let stage_r: u8 = rec
                .get(3)
                .ok_or_else(|| anyhow!("manifest missing stage_r"))?
                .parse()
                .map_err(|_| anyhow!("Invalid stage_r"))?;
            let stage_t: u8 = rec
                .get(4)
                .ok_or_else(|| anyhow!("manifest missing stage_t"))?
                .parse()
                .map_err(|_| anyhow!("Invalid stage_t"))?;
            let stage_s: u8 = rec
                .get(5)
                .ok_or_else(|| anyhow!("manifest missing stage_s"))?
                .parse()
                .map_err(|_| anyhow!("Invalid stage_s"))?;
            let stage_g: u8 = rec
                .get(6)
                .ok_or_else(|| anyhow!("manifest missing stage_g"))?
                .parse()
                .map_err(|_| anyhow!("Invalid stage_g"))?;

            let stage_r = if infected == 0 { 0 } else { stage_r };
            let stage_t = if infected == 0 { 0 } else { stage_t };
            let stage_s = if infected == 0 { 0 } else { stage_s };
            let stage_g = if infected == 0 { 0 } else { stage_g };

            let source_image_id = rec
                .get(7)
                .ok_or_else(|| anyhow!("manifest missing source_image_id"))?
                .to_string();

            items.push(MpIdbItem {
                image_path: crop_path,
                infected,
                species_id,
                stage_labels: [stage_r, stage_t, stage_s, stage_g],
                source_image_id,
            });
        }

        let dataset = Self {
            items,
            cache: None,
            target_height,
            target_width,
            use_cache,
        };

        let dataset = if use_cache {
            let cache = dataset.build_cache();
            Self {
                cache: Some(Arc::new(cache)),
                ..dataset
            }
        } else {
            dataset
        };

        Ok(dataset)
    }

    fn build_cache(&self) -> HashMap<PathBuf, Vec<f32>> {
        let mut cache = HashMap::with_capacity(self.items.len());
        for item in &self.items {
            let p = PathBuf::from(&item.image_path);
            if let Ok(data) = MalariaDataset::load_and_preprocess_image_raw(
                &p,
                self.target_height,
                self.target_width,
            ) {
                cache.insert(p, data);
            }
        }
        cache
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn split_by_source(&self, ratio: f32, seed: u64) -> (Self, Self) {
        assert!(ratio > 0.0 && ratio < 1.0, "Ratio must be between 0 and 1");

        let mut sources: Vec<String> = self
            .items
            .iter()
            .map(|i| i.source_image_id.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let mut rng = StdRng::seed_from_u64(seed);
        sources.shuffle(&mut rng);

        let split_index = (sources.len() as f32 * ratio) as usize;
        let train_sources: HashSet<String> = sources[..split_index].iter().cloned().collect();

        let mut train_items = Vec::new();
        let mut valid_items = Vec::new();
        for it in &self.items {
            if train_sources.contains(&it.source_image_id) {
                train_items.push(it.clone());
            } else {
                valid_items.push(it.clone());
            }
        }

        (
            Self {
                items: train_items,
                cache: self.cache.clone(),
                target_height: self.target_height,
                target_width: self.target_width,
                use_cache: self.use_cache,
            },
            Self {
                items: valid_items,
                cache: self.cache.clone(),
                target_height: self.target_height,
                target_width: self.target_width,
                use_cache: self.use_cache,
            },
        )
    }

    pub fn get(&self, index: usize) -> Option<MpIdbItem> {
        self.items.get(index).cloned()
    }
}

impl Dataset<MpIdbItem> for MpIdbDataset {
    fn get(&self, index: usize) -> Option<MpIdbItem> {
        self.get(index)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

fn species_to_id(species: &str) -> Option<u8> {
    match species {
        "Falciparum" => Some(0),
        "Malariae" => Some(1),
        "Ovale" => Some(2),
        "Vivax" => Some(3),
        "Uninfected" => Some(4),
        _ => None,
    }
}

/// Batch for the malaria model
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MalariaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MpIdbBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub infected: Tensor<B, 1, Int>,
    pub species: Tensor<B, 1, Int>,
    pub stages: Tensor<B, 2>,
}

/// ✅ FIXED BATCHER - GOLDEN RULE RESPECTED
#[allow(dead_code)]
pub struct MalariaBatcher<B: Backend> {
    pub image_height: usize,
    pub image_width: usize,
    _phantom: std::marker::PhantomData<B>,
}

#[allow(dead_code)]
impl<B: Backend> MalariaBatcher<B> {
    pub fn new(image_height: usize, image_width: usize) -> Self {
        Self {
            image_height,
            image_width,
            _phantom: std::marker::PhantomData,
        }
    }
}

// ✅ FIX: Batcher takes 3 generic arguments: B (Backend), I (Input), O (Output)
impl<B: Backend> Batcher<B, MalariaItem, MalariaBatch<B>> for MalariaBatcher<B> {
    /// ✅ CRITICAL FIX: Use ONLY the provided device
    fn batch(&self, items: Vec<MalariaItem>, device: &B::Device) -> MalariaBatch<B> {
        let batch_size = items.len();
        let expected_size = batch_size * 3 * self.image_height * self.image_width;

        let mut images_data = Vec::with_capacity(expected_size);
        let mut labels_data = Vec::with_capacity(batch_size);

        let default_image = vec![0.0; self.image_height * self.image_width * 3];

        // ✅ Preprocessing on CPU
        for item in items {
            let image_data = match MalariaDataset::load_and_preprocess_image_raw(
                Path::new(&item.image_path),
                self.image_height,
                self.image_width,
            ) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("⚠️  Error loading {}: {}", item.image_path, e);
                    default_image.clone()
                }
            };
            images_data.extend(image_data);
            labels_data.push(item.label as i64);
        }

        // ✅ CRITICAL SIZE CHECK (wgpu may crash without this)
        assert_eq!(
            images_data.len(),
            expected_size,
            "❌ Taille invalide: {} != {}",
            images_data.len(),
            expected_size
        );

        // ✅ GPU TRANSFER with the provided device (NEVER use Device::default())
        let images_tensor_1d = Tensor::<B, 1>::from_floats(images_data.as_slice(), device);
        let images_tensor =
            images_tensor_1d.reshape([batch_size, 3, self.image_height, self.image_width]);
        let labels_tensor = Tensor::<B, 1, Int>::from_ints(labels_data.as_slice(), device);

        MalariaBatch {
            images: images_tensor,
            labels: labels_tensor,
        }
    }
}

pub struct MpIdbBatcher<B: Backend> {
    pub image_height: usize,
    pub image_width: usize,
    _phantom: std::marker::PhantomData<B>,
}

#[allow(dead_code)]
impl<B: Backend> MpIdbBatcher<B> {
    pub fn new(image_height: usize, image_width: usize) -> Self {
        Self {
            image_height,
            image_width,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, MpIdbItem, MpIdbBatch<B>> for MpIdbBatcher<B> {
    fn batch(&self, items: Vec<MpIdbItem>, device: &B::Device) -> MpIdbBatch<B> {
        let batch_size = items.len();
        let expected_size = batch_size * 3 * self.image_height * self.image_width;

        let mut images_data = Vec::with_capacity(expected_size);
        let mut infected_data = Vec::with_capacity(batch_size);
        let mut species_data = Vec::with_capacity(batch_size);
        let mut stages_data = Vec::with_capacity(batch_size * 4);

        let default_image = vec![0.0; self.image_height * self.image_width * 3];

        for item in items {
            let image_data = match MalariaDataset::load_and_preprocess_image_raw(
                Path::new(&item.image_path),
                self.image_height,
                self.image_width,
            ) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("⚠️  Error loading {}: {}", item.image_path, e);
                    default_image.clone()
                }
            };
            images_data.extend(image_data);
            infected_data.push(item.infected as i64);
            species_data.push(item.species_id as i64);

            for v in item.stage_labels {
                stages_data.push(v as f32);
            }
        }

        assert_eq!(
            images_data.len(),
            expected_size,
            "❌ Taille invalide: {} != {}",
            images_data.len(),
            expected_size
        );

        let images_tensor_1d = Tensor::<B, 1>::from_floats(images_data.as_slice(), device);
        let images_tensor =
            images_tensor_1d.reshape([batch_size, 3, self.image_height, self.image_width]);

        let species_tensor = Tensor::<B, 1, Int>::from_ints(species_data.as_slice(), device);
        let infected_tensor = Tensor::<B, 1, Int>::from_ints(infected_data.as_slice(), device);
        let stages_tensor =
            Tensor::<B, 1>::from_floats(stages_data.as_slice(), device).reshape([batch_size, 4]);

        MpIdbBatch {
            images: images_tensor,
            infected: infected_tensor,
            species: species_tensor,
            stages: stages_tensor,
        }
    }
}
