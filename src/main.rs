use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Write},
    iter::zip,
    path::Path,
};

use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageReader};
// use oklab::{Oklab, srgb_to_oklab};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The input image
    image: String,

    /// Thumbnail directory
    #[arg(short, long, default_value_t = String::from("./thumbnails/"))]
    thumbs: String,

    /// Size of the thumbnails
    #[arg(short = 'T', long, default_value_t = 32)]
    thumbsize: u32,

    /// Sampling resolution of image thumbnails
    #[arg(short, long, default_value_t = 3)]
    sampleres: u32,

    /// Resolution multiplier for final image
    #[arg(short, long, default_value_t = 2)]
    dpr: u32,
}

#[derive(Serialize, Deserialize)]
struct ThumbnailData {
    path: String,
    res: u32,
    colors: Vec<[f32; 3]>,
}

#[derive(Default, Serialize, Deserialize)]
struct ThumbnailDb {
    thumbs: Vec<ThumbnailData>,
}

async fn main() {
    let args = Args::parse();
    println!("Targeting {}!", args.image);

    let mut thumbs_db = ThumbnailDb::default();
    let mut thumbs_cache: HashMap<String, DynamicImage> = HashMap::new();
    let thumb_data_path = Path::new(args.thumbs.as_str()).with_file_name("thumbdata");

    // Load thumbnail data from cache
    if let Ok(thumb_data) = fs::read(&thumb_data_path) {
        // Read thumb data
        thumbs_db = ron::de::from_bytes(&thumb_data).expect("to deserialize thumbdata");
    }

    println!(
        "Loaded data for {} thumbs from {}!",
        &thumbs_db.thumbs.len(),
        &thumb_data_path.to_str().expect("to convert path to string")
    );

    let mut dirty_thumbs_db = 0u32;

    let mut dir = fs::read_dir(args.thumbs)
        .expect("to read thumbs directory")
        .into_iter();
    while let Some(Ok(thumb_entry)) = dir.next() {
        let entry_path = String::from(
            thumb_entry
                .path()
                .to_str()
                .expect("to convert thumbnail path into a string"),
        );

        if thumbs_db
            .thumbs
            .iter()
            .find(|&a| (a.path == entry_path) && (a.res == args.sampleres))
            .is_none()
        {
            print!("\rProcessing new thumb {:?}", thumb_entry.path());
            std::io::stdout().flush().unwrap(); // Ensure stdout is flushed
            import_thumb(
                thumb_entry.path().to_str().unwrap(),
                args.sampleres,
                Some(&mut thumbs_db),
                &mut thumbs_cache,
            )
            .expect("to import thumbnail");
            dirty_thumbs_db += 1;
        }
    }

    if dirty_thumbs_db > 0 {
        fs::write(
            thumb_data_path,
            ron::ser::to_string_pretty(&thumbs_db, ron::ser::PrettyConfig::default()).unwrap(),
        )
        .expect("to write thumbdata");
        println!(
            "Processed {} new thumbs!                                                  ",
            dirty_thumbs_db
        );
    }

    // Load the target image
    let raw_image = fs::read(&args.image).expect("to read the provided image file");

    let reader = ImageReader::new(Cursor::new(raw_image))
        .with_guessed_format()
        .expect("Cursor io never fails");

    let mut output_path = args.image.clone();
    output_path.insert_str(args.image.rfind(".").unwrap_or(args.image.len()), ".output");
    output_path.push_str(".webp");

    let mut image = reader.decode().expect("to decode image");

    // Crop the image with centre gravity to nearest multiple of thumbsize
    let (width, height) = image.dimensions();

    let crop_width = width - width % args.thumbsize;
    let crop_height = height - height % args.thumbsize;

    image = image.crop(
        (width - crop_width) / 2,
        (height - crop_height) / 2,
        crop_width,
        crop_height,
    );

    // Transform image to linear sRGB
    // image
    //     .apply_color_space(image::metadata::Cicp::SRGB_LINEAR, Default::default())
    //     .expect("to transform color space");

    let image = image.into_rgb8();
    let mut target_image = image::RgbImage::new(crop_width * args.dpr, crop_height * args.dpr);

    let x_chunks = crop_width / args.thumbsize;
    let y_chunks = crop_height / args.thumbsize;
    let chunks = x_chunks * y_chunks;

    // let mut chunk_refs: Vec<&ThumbnailData> = Vec::with_capacity(chunks as usize);
    // let mut result_img = RgbImage::new(crop_width, crop_height);

    for x_chunk in 0..x_chunks {
        for y_chunk in 0..y_chunks {
            print!("\rProcessing {}/{}", (x_chunk * y_chunks) + y_chunk, chunks);
            std::io::stdout().flush().unwrap(); // Ensure stdout is flushed

            let chunk: &DynamicImage = &image
                .view(
                    x_chunk * args.thumbsize,
                    y_chunk * args.thumbsize,
                    args.thumbsize,
                    args.thumbsize,
                )
                .to_image()
                .into();

            let mut thumb = chunk
                .thumbnail_exact(args.sampleres, args.sampleres)
                .to_rgb8();
            let thumb_oklab = rgb_thumb_to_oklab(&mut thumb);

            let mut best_score: f32 = f32::MAX;
            let mut best_match: Option<&ThumbnailData> = None;

            for ref_thumb in &mut thumbs_db.thumbs {
                let score = compare_thumbs(&thumb_oklab, &ref_thumb.colors);

                if score < best_score {
                    best_score = score;
                    best_match = Some(ref_thumb);
                }
            }

            if let Some(best) = best_match {
                if !thumbs_cache.contains_key(&best.path) {
                    import_thumb(&best.path, args.sampleres, None, &mut thumbs_cache)
                        .expect("to import thumb");
                }

                let best_image = thumbs_cache
                    .get(&best.path)
                    .unwrap()
                    .resize(
                        args.thumbsize * args.dpr,
                        args.thumbsize * args.dpr,
                        image::imageops::FilterType::CatmullRom,
                    )
                    .to_rgb8();

                let x = (x_chunk * args.thumbsize * args.dpr) as i64;
                let y = (y_chunk * args.thumbsize * args.dpr) as i64;

                image::imageops::overlay(&mut target_image, &best_image, x, y);

                println!(
                    " - Overlayed {:?} with score {} at {} / {}",
                    Path::new(&best.path).file_name().unwrap(),
                    best_score,
                    x,
                    y
                );
            }
        }
    }
    println!("\rProcessing ............ Done!\n");

    // for chunk in chunk_refs {
    //     println!("{}", chunk.path);
    // }

    // image.to_color_space(image::metadata::Cicp::SRGB, Default::default());

    // image
    //     .apply_color_space(image::metadata::Cicp::SRGB, Default::default())
    //     .expect("to transform output image color space");

    target_image
        .save_with_format(&output_path, image::ImageFormat::WebP)
        .expect("to save output image");
}

fn import_thumb<P>(
    p: P,
    res: u32,
    thumbs_db: Option<&mut ThumbnailDb>,
    thumbs_cache: &mut HashMap<String, DynamicImage>,
) -> Result<(), ()>
where
    P: AsRef<std::path::Path> + Into<String>,
{
    let image = load_image(&p);
    let mut thumb_image = get_thumb(&image, res);

    let path = String::from(p.into());

    if let Some(tdb) = thumbs_db {
        tdb.thumbs.push(ThumbnailData {
            path: path.clone(),
            res,
            colors: rgb_thumb_to_oklab(&mut thumb_image),
        });
    }

    thumbs_cache.insert(path, image);
    Ok(())
}

fn get_thumb(image: &DynamicImage, res: u32) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    // image.clone().thumbnail_exact(res, res).to_rgb8()

    image
        .clone()
        .resize_exact(res, res, image::imageops::FilterType::CatmullRom)
        .to_rgb8()
}

fn load_image<P>(p: P) -> DynamicImage
where
    P: AsRef<std::path::Path>,
{
    let raw_image = fs::read(p).expect("to read the provided thumb file");

    let reader = ImageReader::new(Cursor::new(raw_image))
        .with_guessed_format()
        .expect("Cursor io never fails");

    let image = reader.decode().expect("to decode image");
    // image
    //     .apply_color_space(image::metadata::Cicp::SRGB_LINEAR, Default::default())
    //     .expect("to transform color space");

    image
}

fn rgb_thumb_to_oklab(thumb: &mut image::RgbImage) -> Vec<[f32; 3]> {
    // Vec::from_iter(thumb.enumerate_pixels_mut().map(|(_x, _y, pixel)| {
    //     let Oklab { l, a, b } = srgb_to_oklab(oklab::Rgb::from(pixel.0));
    //     [l, a, b]
    // }))

    Vec::from_iter(thumb.enumerate_pixels_mut().map(|(_x, _y, pixel)| {
        // let Oklab { l, a, b } = srgb_to_oklab(oklab::Rgb::from(pixel.0));
        [pixel.0[0] as f32, pixel.0[1] as f32, pixel.0[2] as f32]
    }))
}

fn compare_thumbs(a: &Vec<[f32; 3]>, b: &Vec<[f32; 3]>) -> f32 {
    if !a.len() == b.len() {
        return f32::MAX;
    }

    let mut diff: f32 = 0f32;

    for (x, y) in zip(a, b) {
        diff += (x[0] - y[0]).powf(2f32) + (x[1] - y[1]).powf(2f32) + (x[2] - y[2]).powf(2f32);
    }

    diff
}
