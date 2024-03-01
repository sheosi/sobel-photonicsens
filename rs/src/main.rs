use std::env;

use image::{self, DynamicImage};

mod sobel;

use sobel::Engine;

#[cfg(test)]
mod test;

fn main() {

    let args = env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        println!("Usage: sobel-photonicsens input output");
        return;
    }

    // Nota: las imágenes se pueden reescribir
    let img = image::open(&args[1]).unwrap();
    let img = img.grayscale();
    let img = img.as_luma8().unwrap();
    let engine = sobel::SoftwareEngine::new();
    let img_out: DynamicImage = engine.apply(&img).into();
    img_out.save(&args[2]).unwrap(); // this step is optional but convenient for testing
}
