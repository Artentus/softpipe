use crate::texture::{Texel, Texture};
use slender_math::v2f;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureWrap {
    Clamp,
    Repeat,
    Mirror,
}

impl TextureWrap {
    fn wrap(&self, c: i32, max: u32) -> u32 {
        let max = max as i32;

        match self {
            TextureWrap::Clamp => c.clamp(0, max - 1) as u32,
            TextureWrap::Repeat => c.rem_euclid(max) as u32,
            TextureWrap::Mirror => {
                let c = c.rem_euclid(max * 2);
                if c >= max {
                    ((max * 2) - c) as u32
                } else {
                    c as u32
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Sampler {
    pub min_filter: TextureFilter,
    pub mag_filter: TextureFilter,
    pub mip_filter: TextureFilter,
    pub wrap_x: TextureWrap,
    pub wrap_y: TextureWrap,
}

impl Sampler {
    #[inline]
    pub const fn new(filter: TextureFilter, wrap: TextureWrap) -> Self {
        Self {
            min_filter: filter,
            mag_filter: filter,
            mip_filter: filter,
            wrap_x: wrap,
            wrap_y: wrap,
        }
    }

    fn sample_level<T: Texture>(
        &self,
        texture: &T,
        uv: v2f,
        level: u32,
        filter: TextureFilter,
    ) -> T::Texel {
        let level = level.min(texture.mip_map_levels() - 1);

        let scaled_uv = uv
            * v2f::new(
                (texture.width() >> level) as f32,
                (texture.height() >> level) as f32,
            );

        match filter {
            TextureFilter::Nearest => {
                let u = scaled_uv.x().round();
                let v = scaled_uv.y().round();

                let x = self.wrap_x.wrap(u as i32, texture.width() >> level);
                let y = self.wrap_y.wrap(v as i32, texture.height() >> level);

                texture.get_texel(x, y, level)
            }
            TextureFilter::Linear => {
                let floor_uv = scaled_uv.floor();
                let ceil_uv = scaled_uv.ceil();
                let fract_uv = scaled_uv - floor_uv;

                let min_x = self
                    .wrap_x
                    .wrap(floor_uv.x() as i32, texture.width() >> level);
                let max_x = self
                    .wrap_x
                    .wrap(ceil_uv.x() as i32, texture.width() >> level);
                let min_y = self
                    .wrap_y
                    .wrap(floor_uv.y() as i32, texture.height() >> level);
                let max_y = self
                    .wrap_y
                    .wrap(ceil_uv.y() as i32, texture.height() >> level);

                let t00 = texture.get_texel(min_x, min_y, level);
                let t01 = texture.get_texel(min_x, max_y, level);
                let t10 = texture.get_texel(max_x, min_y, level);
                let t11 = texture.get_texel(max_x, max_y, level);

                let t0 = t00.lerp(t01, fract_uv.y());
                let t1 = t10.lerp(t11, fract_uv.y());

                t0.lerp(t1, fract_uv.x())
            }
        }
    }

    pub fn sample<T: Texture>(&self, texture: &T, uv: v2f, uv_dx: v2f, uv_dy: v2f) -> T::Texel {
        let texture_size = v2f::new(texture.width() as f32, texture.height() as f32);
        let duv = uv_dx.abs().max(uv_dy.abs()) * texture_size;
        let duv = duv.x().max(duv.y());

        let (filter, duv) = if duv <= 1.0 {
            (self.mag_filter, 1.0)
        } else {
            (self.min_filter, duv)
        };

        match self.mip_filter {
            TextureFilter::Nearest => {
                let level = (duv.ceil() as u32).ilog2();
                self.sample_level(texture, uv, level, filter)
            }
            TextureFilter::Linear => {
                let level = duv.log2();
                let level_floor = level.floor() as u32;
                let level_ceil = level.ceil() as u32;

                let texel_floor = self.sample_level(texture, uv, level_floor, filter);
                let texel_ceil = self.sample_level(texture, uv, level_ceil, filter);
                texel_floor.lerp(texel_ceil, level.fract())
            }
        }
    }
}
