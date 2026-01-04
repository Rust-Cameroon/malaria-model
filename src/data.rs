use anyhow::{Result, anyhow};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use image::{ImageReader, imageops::FilterType};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::{Path, PathBuf}, sync::Arc};

/// Élément du dataset avec chemin d'image et label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MalariaItem {
    pub image_path: String,
    pub label: u8,
}

/// Dataset pour la détection du paludisme
#[derive(Debug, Clone)]
pub struct MalariaDataset {
    pub images: Vec<PathBuf>,
    pub labels: Vec<u8>,
    pub cache: Option<Arc<HashMap<PathBuf, Vec<f32>>>>,
    pub target_height: usize,
    pub target_width: usize,
    pub use_cache: bool,
}

impl MalariaDataset {
    pub fn new<P: AsRef<Path>>(
        root_dir: P,
        target_height: usize,
        target_width: usize,
        use_cache: bool,
    ) -> Result<Self> {
        let root_dir = root_dir.as_ref();
        println!("📂 Chargement du dataset depuis: {}", root_dir.display());

        let parasitized_dir = root_dir.join("Parasitized");
        let uninfected_dir = root_dir.join("Uninfected");

        if !parasitized_dir.exists() {
            return Err(anyhow!(
                "Dossier manquant: {}/Parasitized/",
                root_dir.display()
            ));
        }
        if !uninfected_dir.exists() {
            return Err(anyhow!(
                "Dossier manquant: {}/Uninfected/",
                root_dir.display()
            ));
        }

        let mut images = Vec::new();
        let mut labels = Vec::new();

        Self::load_images_from_dir(&parasitized_dir, 1, &mut images, &mut labels)?;
        Self::load_images_from_dir(&uninfected_dir, 0, &mut images, &mut labels)?;

        let mut rng = StdRng::seed_from_u64(42);
        let mut combined: Vec<_> = images.into_iter().zip(labels.into_iter()).collect();
        combined.shuffle(&mut rng);
        let (shuffled_images, shuffled_labels): (Vec<_>, Vec<_>) = combined.into_iter().unzip();

        println!("📊 Dataset chargé: {} images au total", shuffled_images.len());
        println!("   - Parasitized: {}", shuffled_labels.iter().filter(|&&l| l == 1).count());
        println!("   - Uninfected: {}", shuffled_labels.iter().filter(|&&l| l == 0).count());

        let dataset = Self {
            images: shuffled_images,
            labels: shuffled_labels,
            cache: None,
            target_height,
            target_width,
            use_cache,
        };

        let dataset = if use_cache {
            println!("💾 Initialisation du cache...");
            let cache = dataset.build_cache();
            println!("✅ Cache initialisé avec {} images.", cache.len());
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
            if let Ok(data) = Self::load_and_preprocess_image_raw(path, self.target_height, self.target_width) {
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
            .map_err(|e| anyhow!("Erreur lecture dossier {}: {}", dir.display(), e))?;

        let mut count = 0;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp") {
                        images.push(path);
                        labels.push(label);
                        count += 1;
                    }
                }
            }
        }
        println!("   - {} : {} images chargées", dir.display(), count);
        Ok(())
    }

    /// ✅ CPU SEULEMENT - Prétraitement optimisé
    pub fn load_and_preprocess_image_raw(
        path: &Path,
        target_height: usize,
        target_width: usize,
    ) -> Result<Vec<f32>> {
        let img = ImageReader::open(path)?
            .decode()?
            .resize_exact(
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
            return Err(anyhow!("Index {} hors limites", index));
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
        assert!(ratio > 0.0 && ratio < 1.0, "Le ratio doit être entre 0 et 1");
        let split_index = (self.images.len() as f32 * ratio) as usize;

        let train_images = self.images[..split_index].to_vec();
        let train_labels = self.labels[..split_index].to_vec();
        let valid_images = self.images[split_index..].to_vec();
        let valid_labels = self.labels[split_index..].to_vec();

        println!("📈 Split du dataset (ratio: {}):", ratio);
        println!("   - Entraînement: {} images", train_images.len());
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

/// Batch pour le modèle de paludisme
#[derive(Debug, Clone)]
pub struct MalariaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

/// ✅ BATCHER CORRIGÉ - RÈGLE D'OR RESPECTÉE
pub struct MalariaBatcher<B: Backend> {
    pub image_height: usize,
    pub image_width: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> MalariaBatcher<B> {
    pub fn new(image_height: usize, image_width: usize) -> Self {
        Self {
            image_height,
            image_width,
            _phantom: std::marker::PhantomData,
        }
    }
}

// ✅ CORRECTION : Batcher prend 3 arguments génériques : B (Backend), I (Input), O (Output)
impl<B: Backend> Batcher<B, MalariaItem, MalariaBatch<B>> for MalariaBatcher<B> {
    /// ✅ CORRECTION CRITIQUE : Utilise EXCLUSIVEMENT le device fourni
    fn batch(&self, items: Vec<MalariaItem>, device: &B::Device) -> MalariaBatch<B> {
        let batch_size = items.len();
        let expected_size = batch_size * 3 * self.image_height * self.image_width;
        
        let mut images_data = Vec::with_capacity(expected_size);
        let mut labels_data = Vec::with_capacity(batch_size);

        let default_image = vec![0.0; self.image_height * self.image_width * 3];

        // ✅ Prétraitement sur CPU
        for item in items {
            let image_data = match MalariaDataset::load_and_preprocess_image_raw(
                Path::new(&item.image_path),
                self.image_height,
                self.image_width,
            ) {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("⚠️  Erreur chargement {}: {}", item.image_path, e);
                    default_image.clone()
                }
            };
            images_data.extend(image_data);
            labels_data.push(item.label as i64);
        }

        // ✅ VÉRIFICATION TAILLE CRITIQUE (wgpu crash sans ça)
        assert_eq!(
            images_data.len(),
            expected_size,
            "❌ Taille invalide: {} != {}",
            images_data.len(),
            expected_size
        );

        // ✅ TRANSFERT GPU avec le device fourni (JAMAIS Device::default())
        let images_tensor_1d = Tensor::<B, 1>::from_floats(images_data.as_slice(), device);
        let images_tensor = images_tensor_1d.reshape([batch_size, 3, self.image_height, self.image_width]);
        let labels_tensor = Tensor::<B, 1, Int>::from_ints(labels_data.as_slice(), device);

        MalariaBatch {
            images: images_tensor,
            labels: labels_tensor,
        }
    }
}