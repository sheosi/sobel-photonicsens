use image::{ImageBuffer, Luma};

// Interfaz
pub trait Engine {
    fn apply(&self, img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>>;
}

// Crea código para implementar las interfaces de Clone (clonado) y
// Debug(impresión por pantalla)
#[derive(Clone, Debug)]
pub struct SoftwareEngine {}

// Los métodos y funciones estáticas se definen aquí mediente estos bloques
impl SoftwareEngine {
    pub fn new() -> Self {
        Self{}
    }
}

impl SoftwareEngine {
    fn calc_pixel(img: &ImageBuffer<Luma<u8>, Vec<u8>>, x: u32, y: u32) -> u8 {
        assert!(x > 0 && y > 0 && x < (img.width() - 2) && y < (img.height() - 2));

        const KERNELX: [[i16;3];3] = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ];

        const KERNELY: [[i16;3];3] = [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ];

        let mag_x: i16 = (KERNELX[0][0] * img[(x-1,y-1)][0] as i16) + (KERNELX[0][1] * img[(x,y-1)][0] as i16) + (KERNELX[0][2] * img[(x+1,y-1)][0] as i16) +
                (KERNELX[1][0] * img[(x-1,y)][0] as i16)   + (KERNELX[1][1] * img[(x,y)][0] as i16)   + (KERNELX[1][2] * img[(x+1,y)][0] as i16) +
                (KERNELX[2][0] * img[(x-1,y+1)][0] as i16) + (KERNELX[2][1] * img[(x,y+1)][0] as i16) + (KERNELX[2][2] * img[(x+1,y+1)][0] as i16);

        let mag_y: i16 = (KERNELY[0][0] * img[(x-1,y-1)][0] as i16) + (KERNELY[0][1] * img[(x,y-1)][0] as i16) + (KERNELY[0][2] * img[(x+1,y-1)][0] as i16) +
                (KERNELY[2][0] * img[(x-1,y+1)][0] as i16) + (KERNELY[2][1] * img[(x,y+1)][0] as i16) + (KERNELY[2][2] * img[(x+1,y+1)][0] as i16);

        return (((mag_x*mag_x) + (mag_y*mag_y)) as f32).sqrt().ceil() as u8 ;

    }
}

impl Engine for SoftwareEngine {
    fn apply(&self, img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let mut res = ImageBuffer::new(img.width(), img.height());

        (1..img.width()-2).zip(1..img.height()-2).for_each(|(x,y)|{
            let p: &mut Luma<u8> =res.get_pixel_mut(x, y); 
            p[0]= Self::calc_pixel(img, x, y);
        });

        res
    }
}