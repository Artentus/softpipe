#![allow(dead_code)]

use crate::multi_sampled_texture::MultiSampledTexel;
use bytemuck::Pod;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use slender_math::*;
use std::marker::PhantomData;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Replace,
    AlphaTest,
    AlphaBlend,
}

#[inline]
pub fn srgb_to_linear<Color: Texel>(srgb: Color) -> Color {
    // Cubic approximation with <0.2% error:
    // https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html

    const C1: f32 = 0.012522878;
    const C2: f32 = 0.682171111;
    const C3: f32 = 0.305306011;

    srgb * (srgb * (srgb * C3 + C2) + C1)
}

#[inline]
pub fn linear_to_srgb<Color: Texel>(linear: Color) -> Color {
    // Cubic approximation with <0.2% error:
    // https://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html

    const C1: f32 = 0.585122381;
    const C2: f32 = 0.783140355;
    const C3: f32 = 0.368262736;

    let sqrt1 = linear.sqrt();
    let sqrt2 = sqrt1.sqrt();
    let sqrt3 = sqrt2.sqrt();
    (sqrt1 * C1) + (sqrt2 * C2) - (sqrt3 * C3)
}

pub trait Texel:
    Pod
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Add<f32, Output = Self>
    + Sub<f32, Output = Self>
    + Mul<f32, Output = Self>
    + Div<f32, Output = Self>
    + Rem<f32, Output = Self>
    + AddAssign<f32>
    + SubAssign<f32>
    + MulAssign<f32>
    + DivAssign<f32>
    + RemAssign<f32>
{
    type Storage: Pod + Send + Sync;
    type MultiSampled: MultiSampledTexel<Texel = Self>;

    fn lerp(self, rhs: Self, t: f32) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
    fn sqrt(self) -> Self;

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    fn to_storage(self) -> Self::Storage;
    fn from_storage(s: Self::Storage) -> Self;

    fn to_srgb_storage(self) -> Self::Storage {
        let srgb = linear_to_srgb(self);
        srgb.to_storage()
    }

    fn from_srgb_storage(s: Self::Storage) -> Self {
        let srgb = Self::from_storage(s);
        srgb_to_linear(srgb)
    }

    fn blend(self, rhs: Self, mode: BlendMode) -> Option<Self>;
    fn clamp_to_storage_range(self) -> Self;
}

const U8_MAX: f32 = u8::MAX as f32;

impl Texel for f32 {
    type Storage = u8;
    type MultiSampled = [f32; 4];

    #[inline]
    fn lerp(self, rhs: Self, t: f32) -> Self {
        self + ((rhs - self) * t)
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.clamp(min, max)
    }

    #[inline]
    fn to_storage(self) -> Self::Storage {
        (self * U8_MAX) as u8
    }

    #[inline]
    fn from_storage(s: Self::Storage) -> Self {
        (s as f32) / U8_MAX
    }

    #[inline]
    fn blend(self, _rhs: Self, _mode: BlendMode) -> Option<Self> {
        Some(self)
    }

    #[inline]
    fn clamp_to_storage_range(self) -> Self {
        self.clamp(0.0, 1.0)
    }
}

impl Texel for v2f {
    type Storage = u16;
    type MultiSampled = [v2f; 4];

    #[inline]
    fn lerp(self, rhs: Self, t: f32) -> Self {
        self.lerp(rhs, t)
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline]
    fn sqrt(self) -> Self {
        v2f::new(self.x().sqrt(), self.y().sqrt())
    }

    fn to_storage(self) -> Self::Storage {
        let r = (self.r() * U8_MAX) as u8;
        let g = (self.g() * U8_MAX) as u8;
        u16::from_le_bytes([r, g])
    }

    fn from_storage(s: Self::Storage) -> Self {
        let bytes = s.to_le_bytes();
        let r = (bytes[0] as f32) / U8_MAX;
        let g = (bytes[1] as f32) / U8_MAX;
        v2f::new(r, g)
    }

    #[inline]
    fn blend(self, _rhs: Self, _mode: BlendMode) -> Option<Self> {
        Some(self)
    }

    #[inline]
    fn clamp_to_storage_range(self) -> Self {
        self.clamp(v2f::ZERO, v2f::ONE)
    }
}

impl Texel for v4f {
    type Storage = u32;
    type MultiSampled = [v4f; 4];

    #[inline]
    fn lerp(self, rhs: Self, t: f32) -> Self {
        self.lerp(rhs, t)
    }

    #[inline]
    fn min(self, rhs: Self) -> Self {
        self.min(rhs)
    }

    #[inline]
    fn max(self, rhs: Self) -> Self {
        self.max(rhs)
    }

    #[inline]
    fn sqrt(self) -> Self {
        v4f::new(
            self.x().sqrt(),
            self.y().sqrt(),
            self.z().sqrt(),
            self.w().sqrt(),
        )
    }

    fn to_storage(self) -> Self::Storage {
        let r = (self.r() * U8_MAX) as u8;
        let g = (self.g() * U8_MAX) as u8;
        let b = (self.b() * U8_MAX) as u8;
        let a = (self.a() * U8_MAX) as u8;
        u32::from_le_bytes([r, g, b, a])
    }

    fn from_storage(s: Self::Storage) -> Self {
        let bytes = s.to_le_bytes();
        let r = (bytes[0] as f32) / U8_MAX;
        let g = (bytes[1] as f32) / U8_MAX;
        let b = (bytes[2] as f32) / U8_MAX;
        let a = (bytes[3] as f32) / U8_MAX;
        v4f::new(r, g, b, a)
    }

    fn to_srgb_storage(self) -> Self::Storage {
        let mut srgb = linear_to_srgb(self);
        *srgb.a_mut() = self.a();
        srgb.to_storage()
    }

    fn from_srgb_storage(s: Self::Storage) -> Self {
        let srgb = Self::from_storage(s);
        let mut linear = srgb_to_linear(srgb);
        *linear.a_mut() = srgb.a();
        linear
    }

    #[inline]
    fn blend(self, rhs: Self, mode: BlendMode) -> Option<Self> {
        match mode {
            BlendMode::Replace => Some(self),
            BlendMode::AlphaTest => {
                if self.a() >= 0.5 {
                    Some(self)
                } else {
                    None
                }
            }
            BlendMode::AlphaBlend => {
                if self.a() > 0.0 {
                    Some(rhs.lerp(self, self.a()))
                } else {
                    None
                }
            }
        }
    }

    #[inline]
    fn clamp_to_storage_range(self) -> Self {
        self.clamp(v4f::ZERO, v4f::ONE)
    }
}

pub trait ChannelDesc {
    type Texel: Texel;
}

pub enum Channels<const N: usize> {}

impl ChannelDesc for Channels<1> {
    type Texel = f32;
}

impl ChannelDesc for Channels<2> {
    type Texel = v2f;
}

impl ChannelDesc for Channels<4> {
    type Texel = v4f;
}

pub trait TextureRow<'a, T: Texel>: Send {
    fn width(&self) -> u32;
    fn get_texel(&self, x: u32) -> T;
    fn set_texel(&mut self, x: u32, t: T);
    fn clear(&mut self, t: T);

    fn blend_texel(&mut self, x: u32, t: T, mode: BlendMode) -> bool {
        let prev = self.get_texel(x);
        if let Some(new) = t.blend(prev, mode) {
            self.set_texel(x, new);
            return true;
        }
        false
    }
}

pub trait Texture {
    type Texel: Texel;
    type Row<'a>: TextureRow<'a, Self::Texel>
    where
        Self: 'a;
    type RowIter<'a>: Iterator<Item = Self::Row<'a>>
    where
        Self: 'a;
    type ParRowIter<'a>: IndexedParallelIterator<Item = Self::Row<'a>>
    where
        Self: 'a;

    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn mip_map_levels(&self) -> u32;
    fn get_texel(&self, x: u32, y: u32, level: u32) -> Self::Texel;
    fn set_texel(&mut self, x: u32, y: u32, t: Self::Texel);
    fn rows_mut(&mut self) -> Self::RowIter<'_>;
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_>;
    fn clear(&mut self, t: Self::Texel);
    fn resize(&mut self, new_width: u32, new_height: u32);
    fn bytes_per_texel(&self) -> u32;
    fn data(&self) -> &[u8];

    fn blend_texel(&mut self, x: u32, y: u32, t: Self::Texel, mode: BlendMode) -> bool {
        let prev = self.get_texel(x, y, 0);
        if let Some(new) = t.blend(prev, mode) {
            self.set_texel(x, y, new);
            return true;
        }
        false
    }
}

pub trait ColorSpace {}

pub enum Linear {}
impl ColorSpace for Linear {}

pub enum Srgb {}
impl ColorSpace for Srgb {}

#[repr(transparent)]
pub struct ColorTextureRow<'a, CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    texels: &'a mut [<<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage],
}

impl<'a, const CHANNELS: usize> TextureRow<'a, <Channels<CHANNELS> as ChannelDesc>::Texel>
    for ColorTextureRow<'a, Linear, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    #[inline]
    fn width(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(&self, x: u32) -> <Channels<CHANNELS> as ChannelDesc>::Texel {
        <Channels<CHANNELS> as ChannelDesc>::Texel::from_storage(self.texels[x as usize])
    }

    #[inline]
    fn set_texel(&mut self, x: u32, t: <Channels<CHANNELS> as ChannelDesc>::Texel) {
        self.texels[x as usize] = t.to_storage()
    }

    #[inline]
    fn clear(&mut self, t: <Channels<CHANNELS> as ChannelDesc>::Texel) {
        self.texels.fill(t.to_storage());
    }
}

impl<'a, const CHANNELS: usize> TextureRow<'a, <Channels<CHANNELS> as ChannelDesc>::Texel>
    for ColorTextureRow<'a, Srgb, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    #[inline]
    fn width(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(&self, x: u32) -> <Channels<CHANNELS> as ChannelDesc>::Texel {
        <Channels<CHANNELS> as ChannelDesc>::Texel::from_srgb_storage(self.texels[x as usize])
    }

    #[inline]
    fn set_texel(&mut self, x: u32, t: <Channels<CHANNELS> as ChannelDesc>::Texel) {
        self.texels[x as usize] = t.to_srgb_storage()
    }

    #[inline]
    fn clear(&mut self, t: <Channels<CHANNELS> as ChannelDesc>::Texel) {
        self.texels.fill(t.to_srgb_storage());
    }
}

#[repr(transparent)]
pub struct ColorTextureRowIter<'a, CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    chunks:
        std::slice::ChunksMut<'a, <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage>,
}

impl<'a, CS: ColorSpace, const CHANNELS: usize> Iterator for ColorTextureRowIter<'a, CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Item = ColorTextureRow<'a, CS, CHANNELS>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next().map(|texels| ColorTextureRow {
            _cs: PhantomData,
            texels,
        })
    }
}

#[repr(transparent)]
pub struct ColorTextureParRowIter<'a, CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    chunks:
        rayon::slice::ChunksMut<'a, <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage>,
}

impl<'a, CS: ColorSpace, const CHANNELS: usize> ParallelIterator
    for ColorTextureParRowIter<'a, CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Item = ColorTextureRow<'a, CS, CHANNELS>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        self.chunks
            .map(|texels| ColorTextureRow {
                _cs: PhantomData,
                texels,
            })
            .drive_unindexed(consumer)
    }
}

impl<'a, CS: ColorSpace, const CHANNELS: usize> IndexedParallelIterator
    for ColorTextureParRowIter<'a, CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.chunks
            .map(|texels| ColorTextureRow {
                _cs: PhantomData,
                texels,
            })
            .drive(consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        self.chunks
            .map(|texels| ColorTextureRow {
                _cs: PhantomData,
                texels,
            })
            .with_producer(callback)
    }
}

pub struct ColorTexture<CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    width: u32,
    height: u32,
    texels: Box<[Box<[<<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage]>]>,
}

impl<CS: ColorSpace, const CHANNELS: usize> ColorTexture<CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    pub fn new(width: u32, height: u32) -> Self {
        let texel_size = (width as usize) * (height as usize);

        Self {
            _cs: PhantomData,
            width,
            height,
            texels: Box::new([unsafe { Box::new_zeroed_slice(texel_size).assume_init() }]),
        }
    }
}

pub trait LoadableTexture: Sized {
    fn load<P: AsRef<Path>>(path: P, generate_mip_maps: bool) -> image::ImageResult<Self>;
}

macro_rules! impl_loadable {
    ($cs:ty, $channels:literal, $f:ident) => {
        impl LoadableTexture for ColorTexture<$cs, $channels> {
            fn load<P: AsRef<Path>>(path: P, generate_mip_maps: bool) -> image::ImageResult<Self> {
                let mut image = image::open(path)?.$f();
                let width = image.width();
                let height = image.height();

                let mip_map_count = if generate_mip_maps {
                    image.width().ilog2().min(image.height().ilog2()) as usize
                } else {
                    1
                };

                let mut mip_maps = Vec::with_capacity(mip_map_count);
                for _ in 0..mip_map_count {
                    let texel_size = (image.width() as usize) * (image.height() as usize);
                    let mut texels = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
                    texels.copy_from_slice(bytemuck::cast_slice(image.as_raw()));
                    mip_maps.push(texels);

                    image = image::imageops::resize(
                        &image,
                        image.width() >> 1,
                        image.height() >> 1,
                        image::imageops::FilterType::Lanczos3,
                    );

                    assert!(image.width() > 0);
                    assert!(image.height() > 0);
                }

                Ok(Self {
                    _cs: PhantomData,
                    width,
                    height,
                    texels: mip_maps.into_boxed_slice(),
                })
            }
        }
    };
}

impl_loadable!(Linear, 1, to_luma8);
impl_loadable!(Linear, 2, to_luma_alpha8);
impl_loadable!(Linear, 4, to_rgba8);

impl_loadable!(Srgb, 1, to_luma8);
impl_loadable!(Srgb, 2, to_luma_alpha8);
impl_loadable!(Srgb, 4, to_rgba8);

impl<const CHANNELS: usize> Texture for ColorTexture<Linear, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Texel = <Channels<CHANNELS> as ChannelDesc>::Texel;
    type Row<'a> = ColorTextureRow<'a, Linear, CHANNELS>;
    type RowIter<'a> = ColorTextureRowIter<'a, Linear, CHANNELS>;
    type ParRowIter<'a> = ColorTextureParRowIter<'a, Linear, CHANNELS>;

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn mip_map_levels(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(&self, x: u32, y: u32, level: u32) -> Self::Texel {
        debug_assert!(level < self.mip_map_levels());
        debug_assert!(x < (self.width >> level));
        debug_assert!(y < (self.height >> level));

        let texel_index = (x as usize) + ((y as usize) * ((self.width >> level) as usize));
        Self::Texel::from_storage(self.texels[level as usize][texel_index])
    }

    #[inline]
    fn set_texel(&mut self, x: u32, y: u32, t: Self::Texel) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        debug_assert_eq!(self.texels.len(), 1);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[0][texel_index] = t.to_storage();
    }

    #[inline]
    fn rows_mut(&mut self) -> Self::RowIter<'_> {
        debug_assert_eq!(self.texels.len(), 1);

        ColorTextureRowIter {
            _cs: PhantomData,
            chunks: self.texels[0].chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_> {
        debug_assert_eq!(self.texels.len(), 1);

        use rayon::prelude::*;
        ColorTextureParRowIter {
            _cs: PhantomData,
            chunks: self.texels[0].par_chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn clear(&mut self, t: Self::Texel) {
        debug_assert_eq!(self.texels.len(), 1);

        self.texels[0].fill(t.to_storage());
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        debug_assert_eq!(self.texels.len(), 1);

        self.width = new_width;
        self.height = new_height;

        let texel_size = (new_width as usize) * (new_height as usize);
        self.texels[0] = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
    }

    #[inline]
    fn bytes_per_texel(&self) -> u32 {
        CHANNELS as u32
    }

    #[inline]
    fn data(&self) -> &[u8] {
        debug_assert_eq!(self.texels.len(), 1);

        bytemuck::cast_slice(&self.texels[0])
    }
}

impl<const CHANNELS: usize> Texture for ColorTexture<Srgb, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Texel = <Channels<CHANNELS> as ChannelDesc>::Texel;
    type Row<'a> = ColorTextureRow<'a, Srgb, CHANNELS>;
    type RowIter<'a> = ColorTextureRowIter<'a, Srgb, CHANNELS>;
    type ParRowIter<'a> = ColorTextureParRowIter<'a, Srgb, CHANNELS>;

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn mip_map_levels(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(&self, x: u32, y: u32, level: u32) -> Self::Texel {
        debug_assert!(level < self.mip_map_levels());
        debug_assert!(x < (self.width >> level));
        debug_assert!(y < (self.height >> level));

        let texel_index = (x as usize) + ((y as usize) * ((self.width >> level) as usize));
        Self::Texel::from_srgb_storage(self.texels[level as usize][texel_index])
    }

    #[inline]
    fn set_texel(&mut self, x: u32, y: u32, t: Self::Texel) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        debug_assert_eq!(self.texels.len(), 1);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[0][texel_index] = t.to_srgb_storage();
    }

    #[inline]
    fn rows_mut(&mut self) -> Self::RowIter<'_> {
        debug_assert_eq!(self.texels.len(), 1);

        ColorTextureRowIter {
            _cs: PhantomData,
            chunks: self.texels[0].chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_> {
        debug_assert_eq!(self.texels.len(), 1);

        use rayon::prelude::*;
        ColorTextureParRowIter {
            _cs: PhantomData,
            chunks: self.texels[0].par_chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn clear(&mut self, t: Self::Texel) {
        debug_assert_eq!(self.texels.len(), 1);

        self.texels[0].fill(t.to_srgb_storage());
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        debug_assert_eq!(self.texels.len(), 1);

        self.width = new_width;
        self.height = new_height;

        let texel_size = (new_width as usize) * (new_height as usize);
        self.texels = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
    }

    #[inline]
    fn bytes_per_texel(&self) -> u32 {
        CHANNELS as u32
    }

    #[inline]
    fn data(&self) -> &[u8] {
        debug_assert_eq!(self.texels.len(), 1);

        bytemuck::cast_slice(&self.texels[0])
    }
}

#[repr(transparent)]
pub struct DepthTextureRow<'a> {
    texels: &'a mut [f32],
}

impl<'a> TextureRow<'a, f32> for DepthTextureRow<'a> {
    #[inline]
    fn width(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(&self, x: u32) -> f32 {
        self.texels[x as usize]
    }

    #[inline]
    fn set_texel(&mut self, x: u32, t: f32) {
        self.texels[x as usize] = t
    }

    #[inline]
    fn clear(&mut self, t: f32) {
        self.texels.fill(t);
    }
}

#[repr(transparent)]
pub struct DepthTextureRowIter<'a> {
    chunks: std::slice::ChunksMut<'a, f32>,
}

impl<'a> Iterator for DepthTextureRowIter<'a> {
    type Item = DepthTextureRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next().map(|texels| DepthTextureRow { texels })
    }
}

#[repr(transparent)]
pub struct DepthTextureParRowIter<'a> {
    chunks: rayon::slice::ChunksMut<'a, f32>,
}

impl<'a> ParallelIterator for DepthTextureParRowIter<'a> {
    type Item = DepthTextureRow<'a>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        self.chunks
            .map(|texels| DepthTextureRow { texels })
            .drive_unindexed(consumer)
    }
}

impl<'a> IndexedParallelIterator for DepthTextureParRowIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.chunks
            .map(|texels| DepthTextureRow { texels })
            .drive(consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        self.chunks
            .map(|texels| DepthTextureRow { texels })
            .with_producer(callback)
    }
}

pub struct DepthTexture {
    width: u32,
    height: u32,
    texels: Box<[f32]>,
}

impl DepthTexture {
    pub fn new(width: u32, height: u32) -> Self {
        let texel_size = (width as usize) * (height as usize);

        Self {
            width,
            height,
            texels: unsafe { Box::new_zeroed_slice(texel_size).assume_init() },
        }
    }
}

impl Texture for DepthTexture {
    type Texel = f32;
    type Row<'a> = DepthTextureRow<'a>;
    type RowIter<'a> = DepthTextureRowIter<'a>;
    type ParRowIter<'a> = DepthTextureParRowIter<'a>;

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn mip_map_levels(&self) -> u32 {
        1
    }

    #[inline]
    fn get_texel(&self, x: u32, y: u32, level: u32) -> Self::Texel {
        debug_assert!(level < self.mip_map_levels());
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[texel_index]
    }

    #[inline]
    fn set_texel(&mut self, x: u32, y: u32, t: Self::Texel) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[texel_index] = t;
    }

    #[inline]
    fn rows_mut(&mut self) -> Self::RowIter<'_> {
        DepthTextureRowIter {
            chunks: self.texels.chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_> {
        use rayon::prelude::*;
        DepthTextureParRowIter {
            chunks: self.texels.par_chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn clear(&mut self, t: Self::Texel) {
        self.texels.fill(t);
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;

        let texel_size = (new_width as usize) * (new_height as usize);
        self.texels = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
    }

    #[inline]
    fn bytes_per_texel(&self) -> u32 {
        4
    }

    #[inline]
    fn data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.texels)
    }
}
