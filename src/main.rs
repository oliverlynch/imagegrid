use std::{
    collections::HashMap,
    fs,
    io::{Cursor, Write},
    iter::zip,
    path::Path,
    process::exit,
    sync::OnceLock,
};

use clap::Parser;
use image::{DynamicImage, GenericImageView, ImageReader, RgbImage};
use oklab::{Oklab, srgb_to_oklab};
use serde::{Deserialize, Serialize};
use tokio::task::{self};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The input image
    image: String,

    /// Thumbnail directory
    #[arg(short, long, default_value_t = String::from("./thumbnails/**/*.jpg"))]
    thumbs: String,

    /// Size of the thumbnail grid in pixels
    #[arg(short = 'T', long, default_value_t = 32)]
    thumbsize: u32,

    /// Sampling resolution of image thumbnails
    #[arg(short, long, default_value_t = 4)]
    sampleres: u32,

    /// Resolution multiplier for final image (warning: multiplies image resolution!)
    #[arg(short, long, default_value_t = 1)]
    dpr: u32,

    /// Which algorithm is used to assign thumbnails
    #[arg(short, long, value_enum, default_value_t = DifferenceFunction::Oklab)]
    algorithm: DifferenceFunction,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum DifferenceFunction {
    /// Fast
    Rgb,
    /// Slower, More Accurate
    Oklab,
}

#[derive(Serialize, Deserialize)]
struct ThumbnailData {
    path: String,
    res: u32,
    colors: Vec<[u8; 3]>,
}

#[derive(Default, Serialize, Deserialize)]
struct ThumbnailDb {
    thumbs: Vec<ThumbnailData>,
}

static INPUT_IMAGE: OnceLock<RgbImage> = OnceLock::new();
static THUMBS_DB: OnceLock<ThumbnailDb> = OnceLock::new();
static COMPARISON_FN: OnceLock<DifferenceFunction> = OnceLock::new();

#[tokio::main]
async fn main() {
    let args = Args::parse();
    COMPARISON_FN.get_or_init(|| args.algorithm);

    println!("Targeting {}!", args.image);
    let mut thumbs_db = ThumbnailDb::default();
    let mut thumbs_cache: HashMap<String, DynamicImage> = HashMap::new();

    let thumb_data_path = std::env::current_dir().unwrap().join("thumbdata");

    // Load thumbnail data from cache
    if let Ok(thumb_data) = fs::read(&thumb_data_path) {
        thumbs_db = ron::de::from_bytes(&thumb_data).expect("to deserialize thumbdata");
    }

    println!(
        "Loaded data for {} thumbs from {:?}!",
        &thumbs_db.thumbs.len(),
        &thumb_data_path
    );

    let mut dirty_thumbs_db = 0u32;

    let mut dir = glob::glob(&args.thumbs).expect("to glob directory");
    while let Some(Ok(thumb_entry)) = dir.next() {
        let entry_path = String::from(
            thumb_entry
                .to_str()
                .expect("to convert thumbnail path into a string"),
        );

        if thumbs_db
            .thumbs
            .iter()
            .find(|&a| (a.path == entry_path) && (a.res == args.sampleres))
            .is_none()
        {
            print!("\rProcessing new thumb {:?}", thumb_entry);
            std::io::stdout().flush().unwrap(); // Ensure stdout is flushed

            import_thumb(&entry_path, args.sampleres, &mut thumbs_db).expect("to import thumbnail");
            dirty_thumbs_db += 1;
        }
    }

    if dirty_thumbs_db > 0 {
        fs::write(thumb_data_path, ron::ser::to_string(&thumbs_db).unwrap())
            .expect("to write thumbdata");
        println!(
            "Processed {} new thumbs!                                                  ",
            dirty_thumbs_db
        );
    }

    if thumbs_db.thumbs.len() < 2 {
        eprintln!("Not enough thumbnails found in {}", &args.thumbs);
        exit(1i32);
    }

    // Lock thumbs_db
    THUMBS_DB.get_or_init(|| thumbs_db);

    // Load the target image
    let raw_image = fs::read(&args.image);

    if let Err(e) = raw_image {
        eprintln!("Error loading image '{}': {}", &args.image, e);
        exit(2i32);
    }
    let raw_image = raw_image.unwrap();

    let reader = ImageReader::new(Cursor::new(raw_image))
        .with_guessed_format()
        .expect("Cursor io never fails");

    // Figure out where we want to write the output image
    let original_path = Path::new(&args.image);

    // let output_dir = original_path.parent().unwrap();
    let output_dir = std::env::current_dir().unwrap();
    let output_name = original_path.file_prefix().unwrap().to_str().unwrap();
    let output_ext = original_path.extension().unwrap();

    let mut output_path = output_dir
        .join(output_name)
        .with_extension("output")
        .with_added_extension(output_ext);

    // If the filename already exists try adding a number until it works
    let mut dup_num = 1u32;
    while output_path.exists() {
        output_path = output_dir
            .join(output_name)
            .with_extension(format!("output-{}", dup_num))
            .with_added_extension(output_ext);
        dup_num += 1;
    }

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

    let image = image.into_rgb8();
    INPUT_IMAGE.get_or_init(|| image);

    let mut target_image = image::RgbImage::new(crop_width * args.dpr, crop_height * args.dpr);

    let x_chunks = crop_width / args.thumbsize;
    let y_chunks = crop_height / args.thumbsize;
    let chunks = x_chunks * y_chunks;
    let mut seen_chunks = 0u32;

    // Create a set of tasks to process chunks async
    let mut tasks = task::JoinSet::new();

    for x_chunk in 0..x_chunks {
        for y_chunk in 0..y_chunks {
            tasks.spawn(async move {
                // Do async work
                let chunk: &RgbImage = &INPUT_IMAGE
                    .get()
                    .unwrap()
                    .view(
                        x_chunk * args.thumbsize,
                        y_chunk * args.thumbsize,
                        args.thumbsize,
                        args.thumbsize,
                    )
                    .to_image();

                return (
                    x_chunk,
                    y_chunk,
                    process_chunk(chunk, args.sampleres, THUMBS_DB.get().unwrap())
                        .await
                        .unwrap(),
                );
            });
        }
    }

    while let Some(res) = tasks.join_next().await {
        let (x, y, best) = res.expect("thread failed :(");

        if !thumbs_cache.contains_key(&best.path) {
            let image = load_image(&best.path).resize_exact(
                args.thumbsize * args.dpr,
                args.thumbsize * args.dpr,
                image::imageops::FilterType::CatmullRom,
            );
            thumbs_cache.insert(best.path.clone(), image);
        }

        let best_image = thumbs_cache.get(&best.path).unwrap().to_rgb8();

        let x = (x * args.thumbsize * args.dpr) as i64;
        let y = (y * args.thumbsize * args.dpr) as i64;

        image::imageops::overlay(&mut target_image, &best_image, x, y);

        seen_chunks += 1;
        print!("\rProcessing {}/{}", seen_chunks, chunks);
        std::io::stdout().flush().unwrap(); // Ensure stdout is flushed
    }

    println!("\rProcessing ............ Done!");
    print!("Saving Image...\r");

    target_image
        .save(&output_path)
        .expect("to save output image");

    println!("Saved image to {}", &output_path.display());
}

fn import_thumb<P>(p: P, res: u32, thumbs_db: &mut ThumbnailDb) -> Result<(), ()>
where
    P: AsRef<std::path::Path> + Into<String>,
{
    let image = load_image(&p);
    let thumb_image = get_thumb(&image, res);

    let path = String::from(p.into());

    thumbs_db.thumbs.push(ThumbnailData {
        path: path.clone(),
        res,
        colors: rgb_thumb_to_pixels(&thumb_image),
    });

    Ok(())
}

fn get_thumb(image: &DynamicImage, res: u32) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
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

    image
}

fn rgb_thumb_to_pixels(thumb: &RgbImage) -> Vec<[u8; 3]> {
    Vec::from_iter(thumb.enumerate_pixels().map(|(_x, _y, pixel)| pixel.0))
}

fn compare_thumbs_u8(a: &Vec<[u8; 3]>, b: &Vec<[u8; 3]>) -> i32 {
    if !a.len() == b.len() {
        return i32::max_value();
    }

    let mut diff = 0i32;

    for (x, y) in zip(a, b) {
        diff += (x[0] as i32 - y[0] as i32).pow(2u32)
            + (x[1] as i32 - y[1] as i32).pow(2u32)
            + (x[2] as i32 - y[2] as i32).pow(2u32);
    }

    diff
}

fn compare_thumbs_f32(a: &Vec<[f32; 3]>, b: &Vec<[f32; 3]>) -> f32 {
    if !a.len() == b.len() {
        return f32::MAX;
    }

    let mut diff = 0f32;

    for (x, y) in zip(a, b) {
        diff += (x[0] - y[0]).powi(2i32) + (x[1] - y[1]).powi(2i32) + (x[2] - y[2]).powi(2i32);
    }

    diff
}

fn compare_thumbs_oklab(a: &Vec<[u8; 3]>, b: &Vec<[u8; 3]>) -> f32 {
    let a_rgb = a
        .into_iter()
        .map(|v| lab_to_f32(srgb_to_oklab(oklab::Rgb::from(*v))));
    let b_rgb = b
        .into_iter()
        .map(|v| lab_to_f32(srgb_to_oklab(oklab::Rgb::from(*v))));

    compare_thumbs_f32(&a_rgb.collect(), &b_rgb.collect())
}

fn lab_to_f32(lab: Oklab) -> [f32; 3] {
    [lab.l, lab.a, lab.b]
}

async fn process_chunk<'a>(
    chunk: &RgbImage,
    sampleres: u32,
    thumbs_db: &'a ThumbnailDb,
) -> Option<&'a ThumbnailData> {
    let mut thumb = DynamicImage::from(chunk.clone())
        .resize_exact(
            sampleres,
            sampleres,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();

    let pixels = rgb_thumb_to_pixels(&mut thumb);
    let mut best_match: Option<&ThumbnailData> = None;

    match COMPARISON_FN.get().unwrap() {
        DifferenceFunction::Oklab => {
            let mut best_score = f32::MAX;

            for ref_thumb in &thumbs_db.thumbs {
                let score = compare_thumbs_oklab(&pixels, &ref_thumb.colors);

                if score < best_score {
                    best_score = score;
                    best_match = Some(ref_thumb);
                }
            }
        }
        DifferenceFunction::Rgb => {
            let mut best_score = i32::MAX;

            for ref_thumb in &thumbs_db.thumbs {
                let score = compare_thumbs_u8(&pixels, &ref_thumb.colors);

                if score < best_score {
                    best_score = score;
                    best_match = Some(ref_thumb);
                }
            }
        }
    }

    best_match
}
