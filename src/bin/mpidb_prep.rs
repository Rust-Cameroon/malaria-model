use std::{
    collections::VecDeque,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use csv::Writer;
use image::{imageops::FilterType, DynamicImage, GrayImage, ImageReader, RgbImage};

#[derive(Debug, Clone, Copy)]
struct Stages {
    r: u8,
    t: u8,
    s: u8,
    g: u8,
}

fn center_square_crop(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let side = w.min(h);
    let x0 = (w - side) / 2;
    let y0 = (h - side) / 2;

    let mut out = RgbImage::new(side, side);
    for yy in 0..side {
        for xx in 0..side {
            let px = img.get_pixel(x0 + xx, y0 + yy);
            out.put_pixel(xx, yy, *px);
        }
    }

    out
}

#[derive(Debug, Clone)]
struct ManifestRow {
    crop_path: String,
    infected: u8,
    species: String,
    stage_r: u8,
    stage_t: u8,
    stage_s: u8,
    stage_g: u8,
    source_image_id: String,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let data_root = args.next().unwrap_or_else(|| "data".to_string());
    let out_root = args.next().unwrap_or_else(|| "mpidb_crops".to_string());
    let crop_size: u32 = args
        .next()
        .as_deref()
        .unwrap_or("128")
        .parse()
        .context("crop_size must be an integer")?;
    let min_mask_area: usize = args
        .next()
        .as_deref()
        .unwrap_or("25")
        .parse()
        .context("min_mask_area must be an integer")?;

    let data_root = PathBuf::from(data_root);
    let out_root = PathBuf::from(out_root);
    fs::create_dir_all(&out_root).context("Failed to create output dir")?;

    let species_dirs = list_species_dirs(&data_root)?;
    if species_dirs.is_empty() {
        return Err(anyhow!(
            "No species directories found in {}",
            data_root.display()
        ));
    }

    let mut rows: Vec<ManifestRow> = Vec::new();
    for species_dir in species_dirs {
        let species = species_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Unknown")
            .to_string();

        let gt_dir = species_dir.join("gt");
        let img_dir = species_dir.join("img");
        if !gt_dir.exists() || !img_dir.exists() {
            continue;
        }

        let out_species_dir = out_root.join(&species);
        fs::create_dir_all(&out_species_dir)
            .with_context(|| format!("Failed to create {}", out_species_dir.display()))?;

        for entry in
            fs::read_dir(&gt_dir).with_context(|| format!("Failed to read {}", gt_dir.display()))?
        {
            let entry = entry?;
            let gt_path = entry.path();
            if !is_image_file(&gt_path) {
                continue;
            }

            let file_name = gt_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("Invalid filename"))?
                .to_string();

            let img_path = img_dir.join(&file_name);
            if !img_path.exists() {
                continue;
            }

            let source_image_id = Path::new(&file_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let stages = parse_stages_from_stem(&source_image_id);

            let rgb = read_rgb(&img_path)
                .with_context(|| format!("Failed reading img {}", img_path.display()))?;
            let mask = read_mask(&gt_path)
                .with_context(|| format!("Failed reading gt {}", gt_path.display()))?;

            let comps = connected_components_bboxes(&mask, min_mask_area);
            if comps.is_empty() {
                continue;
            }

            for (i, (min_x, min_y, max_x, max_y)) in comps.into_iter().enumerate() {
                let crop = crop_and_square_pad(&rgb, min_x, min_y, max_x, max_y, 0.25);
                let crop = DynamicImage::ImageRgb8(crop)
                    .resize_exact(crop_size, crop_size, FilterType::Triangle)
                    .to_rgb8();

                let crop_name = format!("{}_{}.png", source_image_id, i);
                let crop_path = out_species_dir.join(crop_name);
                crop.save(&crop_path)
                    .with_context(|| format!("Failed saving {}", crop_path.display()))?;

                rows.push(ManifestRow {
                    crop_path: crop_path.to_string_lossy().to_string(),
                    infected: 1,
                    species: species.clone(),
                    stage_r: stages.r,
                    stage_t: stages.t,
                    stage_s: stages.s,
                    stage_g: stages.g,
                    source_image_id: source_image_id.clone(),
                });
            }
        }
    }

    // Add uninfected samples (binary gate negatives)
    let uninfected_dir = data_root.join("Uninfected");
    if uninfected_dir.exists() {
        let out_uninfected_dir = out_root.join("Uninfected");
        fs::create_dir_all(&out_uninfected_dir)
            .with_context(|| format!("Failed to create {}", out_uninfected_dir.display()))?;

        for entry in fs::read_dir(&uninfected_dir)
            .with_context(|| format!("Failed to read {}", uninfected_dir.display()))?
        {
            let entry = entry?;
            let img_path = entry.path();
            if !is_image_file(&img_path) {
                continue;
            }

            let source_image_id = img_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("uninfected")
                .to_string();

            let rgb = read_rgb(&img_path)
                .with_context(|| format!("Failed reading uninfected img {}", img_path.display()))?;
            let crop = center_square_crop(&rgb);
            let crop = DynamicImage::ImageRgb8(crop)
                .resize_exact(crop_size, crop_size, FilterType::Triangle)
                .to_rgb8();

            let crop_name = format!("{}_0.png", source_image_id);
            let crop_path = out_uninfected_dir.join(crop_name);
            crop.save(&crop_path)
                .with_context(|| format!("Failed saving {}", crop_path.display()))?;

            rows.push(ManifestRow {
                crop_path: crop_path.to_string_lossy().to_string(),
                infected: 0,
                species: "Uninfected".to_string(),
                stage_r: 0,
                stage_t: 0,
                stage_s: 0,
                stage_g: 0,
                source_image_id,
            });
        }
    }

    let manifest_path = out_root.join("manifest.csv");
    write_manifest(&manifest_path, &rows)?;

    println!("Wrote {} rows to {}", rows.len(), manifest_path.display());

    Ok(())
}

fn list_species_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("Failed to read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff")
        })
        .unwrap_or(false)
}

fn parse_stages_from_stem(stem: &str) -> Stages {
    let mut r = 0;
    let mut t = 0;
    let mut s = 0;
    let mut g = 0;

    for token in stem
        .split(|c: char| ['-', '_', ' '].contains(&c))
        .filter(|t| !t.is_empty())
    {
        match token {
            "R" => r = 1,
            "T" => t = 1,
            "S" => s = 1,
            "G" => g = 1,
            _ => {}
        }
    }

    Stages { r, t, s, g }
}

fn read_rgb(path: &Path) -> Result<RgbImage> {
    let img = ImageReader::open(path)?.decode()?;
    Ok(img.to_rgb8())
}

fn read_mask(path: &Path) -> Result<GrayImage> {
    let img = ImageReader::open(path)?.decode()?;
    Ok(img.to_luma8())
}

fn connected_components_bboxes(mask: &GrayImage, min_area: usize) -> Vec<(u32, u32, u32, u32)> {
    let (w, h) = mask.dimensions();
    let mut visited = vec![false; (w as usize) * (h as usize)];
    let mut out = Vec::new();

    let idx = |x: u32, y: u32, w: u32| -> usize { (y as usize) * (w as usize) + (x as usize) };

    for y in 0..h {
        for x in 0..w {
            let i = idx(x, y, w);
            if visited[i] {
                continue;
            }
            visited[i] = true;

            if mask.get_pixel(x, y)[0] == 0 {
                continue;
            }

            let mut q = VecDeque::new();
            q.push_back((x, y));

            let mut area: usize = 0;
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;

            while let Some((cx, cy)) = q.pop_front() {
                area += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);

                for (nx, ny) in neighbors4(cx, cy, w, h) {
                    let ni = idx(nx, ny, w);
                    if visited[ni] {
                        continue;
                    }
                    visited[ni] = true;
                    if mask.get_pixel(nx, ny)[0] == 0 {
                        continue;
                    }
                    q.push_back((nx, ny));
                }
            }

            if area >= min_area {
                out.push((min_x, min_y, max_x, max_y));
            }
        }
    }

    out
}

fn neighbors4(x: u32, y: u32, w: u32, h: u32) -> [(u32, u32); 4] {
    let left = if x > 0 { (x - 1, y) } else { (x, y) };
    let right = if x + 1 < w { (x + 1, y) } else { (x, y) };
    let up = if y > 0 { (x, y - 1) } else { (x, y) };
    let down = if y + 1 < h { (x, y + 1) } else { (x, y) };
    [left, right, up, down]
}

fn crop_and_square_pad(
    img: &RgbImage,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    pad_frac: f32,
) -> RgbImage {
    let (w, h) = img.dimensions();

    let bb_w = (max_x + 1).saturating_sub(min_x);
    let bb_h = (max_y + 1).saturating_sub(min_y);

    let pad_x = ((bb_w as f32) * pad_frac).ceil() as u32;
    let pad_y = ((bb_h as f32) * pad_frac).ceil() as u32;

    let x0 = min_x.saturating_sub(pad_x);
    let y0 = min_y.saturating_sub(pad_y);
    let x1 = (max_x + pad_x).min(w.saturating_sub(1));
    let y1 = (max_y + pad_y).min(h.saturating_sub(1));

    let crop_w = (x1 + 1).saturating_sub(x0);
    let crop_h = (y1 + 1).saturating_sub(y0);

    let side = crop_w.max(crop_h);

    let mut out = RgbImage::new(side, side);

    let offset_x = (side - crop_w) / 2;
    let offset_y = (side - crop_h) / 2;

    for yy in 0..crop_h {
        for xx in 0..crop_w {
            let px = img.get_pixel(x0 + xx, y0 + yy);
            out.put_pixel(offset_x + xx, offset_y + yy, *px);
        }
    }

    out
}

fn write_manifest(path: &Path, rows: &[ManifestRow]) -> Result<()> {
    let mut wtr =
        Writer::from_path(path).with_context(|| format!("Failed to open {}", path.display()))?;
    wtr.write_record([
        "crop_path",
        "infected",
        "species",
        "stage_r",
        "stage_t",
        "stage_s",
        "stage_g",
        "source_image_id",
    ])?;

    for r in rows {
        wtr.write_record([
            r.crop_path.as_str(),
            &r.infected.to_string(),
            r.species.as_str(),
            &r.stage_r.to_string(),
            &r.stage_t.to_string(),
            &r.stage_s.to_string(),
            &r.stage_g.to_string(),
            r.source_image_id.as_str(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
