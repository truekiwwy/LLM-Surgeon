//! Off-screen rendering for `--screenshot` and the tests: an egui frame is run without a
//! window or a GPU, tessellated, and painted by a small CPU rasterizer (textured triangles
//! with premultiplied-alpha blending, as egui expects), then written as a PNG.
//!
//! The output is close to the real window but not pixel-identical to the OpenGL painter
//! (no multisampling, bilinear sampling only): good for captures and structural checks.

use std::collections::HashMap;
use std::io::BufWriter;
use std::path::Path;

use eframe::egui::{self, epaint, Rect, TextureId, Vec2};

/// A rendered frame and every string it painted, in paint order: the "structure" a golden
/// test pins without depending on pixels or fonts.
pub struct Frame {
    pub image: Image,
    pub texts: Vec<String>,
}

fn collect_texts(shape: &epaint::Shape, out: &mut Vec<String>) {
    match shape {
        epaint::Shape::Text(t) => {
            let text = t.galley.text().trim();
            if !text.is_empty() {
                out.push(text.to_owned());
            }
        }
        epaint::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect_texts(s, out)),
        _ => {}
    }
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    /// RGBA, row by row, top to bottom.
    pub rgba: Vec<u8>,
}

impl Image {
    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        let fail = |e: &dyn std::fmt::Display| format!("{} : {e}", path.display());
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| fail(&e))?;
        }
        let file = std::fs::File::create(path).map_err(|e| fail(&e))?;
        let mut encoder = png::Encoder::new(BufWriter::new(file), self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| fail(&e))?;
        writer.write_image_data(&self.rgba).map_err(|e| fail(&e))
    }

    /// Fraction of pixels that differ from the first one: 0 means a blank frame.
    pub fn ink(&self) -> f32 {
        let first = &self.rgba[..4.min(self.rgba.len())];
        let total = self.rgba.len() / 4;
        let differing = self.rgba.chunks_exact(4).filter(|p| *p != first).count();
        differing as f32 / total.max(1) as f32
    }
}

/// Enough for egui to settle its layout and finish a 0.2 s modal fade.
const PASSES: u32 = 6;

struct Texture {
    size: [usize; 2],
    pixels: Vec<[u8; 4]>,
}

impl Texture {
    fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let [w, h] = self.size;
        let (x, y) = (u * w as f32 - 0.5, v * h as f32 - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |xi: f32, yi: f32| {
            let xi = (xi.max(0.0) as usize).min(w - 1);
            let yi = (yi.max(0.0) as usize).min(h - 1);
            self.pixels[yi * w + xi].map(f32::from)
        };
        let (a, b, c, d) = (
            at(x0, y0),
            at(x0 + 1.0, y0),
            at(x0, y0 + 1.0),
            at(x0 + 1.0, y0 + 1.0),
        );
        std::array::from_fn(|i| {
            let top = a[i] * (1.0 - fx) + b[i] * fx;
            let bottom = c[i] * (1.0 - fx) + d[i] * fx;
            (top * (1.0 - fy) + bottom * fy) / 255.0
        })
    }
}

fn apply_delta(
    textures: &mut HashMap<TextureId, Texture>,
    id: TextureId,
    delta: &epaint::ImageDelta,
) {
    let epaint::ImageData::Color(image) = &delta.image;
    let pixels: Vec<[u8; 4]> = image.pixels.iter().map(|c| c.to_array()).collect();
    match delta.pos {
        None => {
            textures.insert(
                id,
                Texture {
                    size: image.size,
                    pixels,
                },
            );
        }
        Some([x0, y0]) => {
            if let Some(tex) = textures.get_mut(&id) {
                for row in 0..image.size[1] {
                    for col in 0..image.size[0] {
                        let (tx, ty) = (x0 + col, y0 + row);
                        if tx < tex.size[0] && ty < tex.size[1] {
                            tex.pixels[ty * tex.size[0] + tx] = pixels[row * image.size[0] + col];
                        }
                    }
                }
            }
        }
    }
}

fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn paint_mesh(
    out: &mut Image,
    mesh: &epaint::Mesh,
    clip: Rect,
    ppp: f32,
    textures: &HashMap<TextureId, Texture>,
) {
    let texture = textures.get(&mesh.texture_id);
    let (w, h) = (out.width as i32, out.height as i32);
    let clip_min = [
        (clip.min.x * ppp).floor().max(0.0) as i32,
        (clip.min.y * ppp).floor().max(0.0) as i32,
    ];
    let clip_max = [
        (clip.max.x * ppp).ceil().min(w as f32) as i32,
        (clip.max.y * ppp).ceil().min(h as f32) as i32,
    ];
    for tri in mesh.indices.chunks_exact(3) {
        let v = [
            mesh.vertices[tri[0] as usize],
            mesh.vertices[tri[1] as usize],
            mesh.vertices[tri[2] as usize],
        ];
        let p = v.map(|v| [v.pos.x * ppp, v.pos.y * ppp]);
        let area = edge(p[0], p[1], p[2]);
        if area.abs() < 1e-6 {
            continue;
        }
        let min_x =
            (p.iter().map(|q| q[0]).fold(f32::MAX, f32::min).floor() as i32).max(clip_min[0]);
        let max_x =
            (p.iter().map(|q| q[0]).fold(f32::MIN, f32::max).ceil() as i32).min(clip_max[0]);
        let min_y =
            (p.iter().map(|q| q[1]).fold(f32::MAX, f32::min).floor() as i32).max(clip_min[1]);
        let max_y =
            (p.iter().map(|q| q[1]).fold(f32::MIN, f32::max).ceil() as i32).min(clip_max[1]);
        for y in min_y..max_y {
            for x in min_x..max_x {
                let c = [x as f32 + 0.5, y as f32 + 0.5];
                let w0 = edge(p[1], p[2], c) / area;
                let w1 = edge(p[2], p[0], c) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let mix = |f: &dyn Fn(usize) -> f32| w0 * f(0) + w1 * f(1) + w2 * f(2);
                let mut src: [f32; 4] =
                    std::array::from_fn(|i| mix(&|k| f32::from(v[k].color.to_array()[i])) / 255.0);
                if let Some(tex) = texture {
                    let t = tex.sample(mix(&|k| v[k].uv.x), mix(&|k| v[k].uv.y));
                    src = std::array::from_fn(|i| src[i] * t[i]);
                }
                let at = ((y * w + x) * 4) as usize;
                let dst = &mut out.rgba[at..at + 4];
                for i in 0..4 {
                    let blended = src[i] * 255.0 + f32::from(dst[i]) * (1.0 - src[3]);
                    dst[i] = blended.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

/// Runs `run` for [`PASSES`] passes (the first ones lay out text and sizes, the others let
/// fade-in animations finish) at `size` pixels and `ppp` pixels per point, and paints the last.
pub fn render(
    ctx: &egui::Context,
    size: [u32; 2],
    ppp: f32,
    mut run: impl FnMut(&mut egui::Ui),
) -> Frame {
    ctx.set_pixels_per_point(ppp);
    let screen = Rect::from_min_size(
        egui::Pos2::ZERO,
        Vec2::new(size[0] as f32 / ppp, size[1] as f32 / ppp),
    );
    let mut textures = HashMap::new();
    let mut last = None;
    for pass in 0..PASSES {
        let input = egui::RawInput {
            screen_rect: Some(screen),
            time: Some(f64::from(pass) * 0.25),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, &mut run);
        // Taken out and cleared: egui panics when a TexturesDelta is dropped unapplied.
        let mut deltas = std::mem::take(&mut output.textures_delta);
        for (id, list) in deltas.set.drain() {
            for delta in &list {
                apply_delta(&mut textures, id, delta);
            }
        }
        deltas.clear();
        last = Some(output);
    }
    let output = last.expect("at least one pass ran");
    let mut image = Image {
        width: size[0],
        height: size[1],
        rgba: [0, 0, 0, 255].repeat((size[0] * size[1]) as usize),
    };
    let mut texts = Vec::new();
    output
        .shapes
        .iter()
        .for_each(|s| collect_texts(&s.shape, &mut texts));
    for prim in ctx.tessellate(output.shapes, ppp) {
        if let epaint::Primitive::Mesh(mesh) = &prim.primitive {
            paint_mesh(&mut image, mesh, prim.clip_rect, ppp, &textures);
        }
    }
    Frame { image, texts }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_painted_and_encoded() {
        let ctx = egui::Context::default();
        let Frame { image, texts } = render(&ctx, [320, 160], 1.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                ui.heading("Surgeon");
                let _ = ui.button("Ouvrir");
            });
        });
        assert_eq!(texts, ["Surgeon", "Ouvrir"]);
        assert!(
            image.ink() > 0.02,
            "text and widgets leave pixels: {}",
            image.ink()
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.png");
        image.save_png(&path).unwrap();
        assert_eq!(&std::fs::read(&path).unwrap()[..4], b"\x89PNG");
    }
}
