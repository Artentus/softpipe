#![allow(dead_code)]

use crate::texture::*;
use bytemuck::Pod;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::marker::PhantomData;

pub trait MultiSampledTexel: Pod + Send + Sync {
    type Texel: Texel<MultiSampled = Self>;

    fn x0y0(&self) -> Self::Texel;
    fn x0y1(&self) -> Self::Texel;
    fn x1y0(&self) -> Self::Texel;
    fn x1y1(&self) -> Self::Texel;

    fn set_x0y0(&mut self, t: Self::Texel);
    fn set_x0y1(&mut self, t: Self::Texel);
    fn set_x1y0(&mut self, t: Self::Texel);
    fn set_x1y1(&mut self, t: Self::Texel);

    fn blend_x0y0(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel>;
    fn blend_x0y1(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel>;
    fn blend_x1y0(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel>;
    fn blend_x1y1(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel>;

    fn to_storage(self) -> [<Self::Texel as Texel>::Storage; 4];
    fn from_storage(s: [<Self::Texel as Texel>::Storage; 4]) -> Self;

    fn to_srgb_storage(mut self) -> [<Self::Texel as Texel>::Storage; 4] {
        self.set_x0y0(linear_to_srgb(self.x0y0()));
        self.set_x0y1(linear_to_srgb(self.x0y1()));
        self.set_x1y0(linear_to_srgb(self.x1y0()));
        self.set_x1y1(linear_to_srgb(self.x1y1()));
        self.to_storage()
    }

    fn from_srgb_storage(s: [<Self::Texel as Texel>::Storage; 4]) -> Self {
        let mut this = Self::from_storage(s);
        this.set_x0y0(srgb_to_linear(this.x0y0()));
        this.set_x0y1(srgb_to_linear(this.x0y1()));
        this.set_x1y0(srgb_to_linear(this.x1y0()));
        this.set_x1y1(srgb_to_linear(this.x1y1()));
        this
    }

    fn resolve(&self) -> Self::Texel;
}

impl<T: Texel<MultiSampled = [T; 4]>> MultiSampledTexel for [T; 4] {
    type Texel = T;

    #[inline]
    fn x0y0(&self) -> Self::Texel {
        self[0]
    }

    #[inline]
    fn x0y1(&self) -> Self::Texel {
        self[1]
    }

    #[inline]
    fn x1y0(&self) -> Self::Texel {
        self[2]
    }

    #[inline]
    fn x1y1(&self) -> Self::Texel {
        self[3]
    }

    #[inline]
    fn set_x0y0(&mut self, t: Self::Texel) {
        self[0] = t;
    }

    #[inline]
    fn set_x0y1(&mut self, t: Self::Texel) {
        self[1] = t;
    }

    #[inline]
    fn set_x1y0(&mut self, t: Self::Texel) {
        self[2] = t;
    }

    #[inline]
    fn set_x1y1(&mut self, t: Self::Texel) {
        self[3] = t;
    }

    #[inline]
    fn blend_x0y0(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel> {
        self[0].blend(rhs, mode)
    }

    #[inline]
    fn blend_x0y1(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel> {
        self[1].blend(rhs, mode)
    }

    #[inline]
    fn blend_x1y0(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel> {
        self[2].blend(rhs, mode)
    }

    #[inline]
    fn blend_x1y1(&self, rhs: Self::Texel, mode: BlendMode) -> Option<Self::Texel> {
        self[3].blend(rhs, mode)
    }

    fn to_storage(self) -> [<Self::Texel as Texel>::Storage; 4] {
        self.map(<Self::Texel as Texel>::to_storage)
    }

    fn from_storage(s: [<Self::Texel as Texel>::Storage; 4]) -> Self {
        s.map(<Self::Texel as Texel>::from_storage)
    }

    #[inline]
    fn resolve(&self) -> Self::Texel {
        (self[0] + self[1] + self[2] + self[3]) * 0.25
    }
}

pub trait MultiSampledTextureRow<'a, T: Texel>: Send {
    fn width(&self) -> u32;
    fn get_texel(&self, x: u32) -> T::MultiSampled;
    fn set_texel(&mut self, x: u32, t: T::MultiSampled);
    fn clear(&mut self, t: T);
}

pub trait MultiSampledTexture {
    type Resolved: Texture;

    type Row<'a>: MultiSampledTextureRow<'a, <Self::Resolved as Texture>::Texel>
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
    fn get_texel(
        &self,
        x: u32,
        y: u32,
    ) -> <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled;
    fn set_texel(
        &mut self,
        x: u32,
        y: u32,
        t: <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled,
    );
    fn rows_mut(&mut self) -> Self::RowIter<'_>;
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_>;
    fn clear(&mut self, t: <Self::Resolved as Texture>::Texel);
    fn resize(&mut self, new_width: u32, new_height: u32);
    fn resolve_into(&mut self, target: &mut Self::Resolved);
}

#[repr(transparent)]
pub struct MultiSampledColorTextureRow<'a, CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    texels: &'a mut [[<<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage; 4]],
}

impl<'a, const CHANNELS: usize>
    MultiSampledTextureRow<'a, <Channels<CHANNELS> as ChannelDesc>::Texel>
    for MultiSampledColorTextureRow<'a, Linear, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    #[inline]
    fn width(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(
        &self,
        x: u32,
    ) -> <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::MultiSampled {
        <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::MultiSampled::from_storage(
            self.texels[x as usize],
        )
    }

    #[inline]
    fn set_texel(
        &mut self,
        x: u32,
        t: <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::MultiSampled,
    ) {
        self.texels[x as usize] = t.to_storage()
    }

    #[inline]
    fn clear(&mut self, t: <Channels<CHANNELS> as ChannelDesc>::Texel) {
        self.texels.fill([t.to_storage(); 4]);
    }
}

impl<'a, const CHANNELS: usize>
    MultiSampledTextureRow<'a, <Channels<CHANNELS> as ChannelDesc>::Texel>
    for MultiSampledColorTextureRow<'a, Srgb, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    #[inline]
    fn width(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(
        &self,
        x: u32,
    ) -> <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::MultiSampled {
        <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::MultiSampled::from_srgb_storage(
            self.texels[x as usize],
        )
    }

    #[inline]
    fn set_texel(
        &mut self,
        x: u32,
        t: <<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::MultiSampled,
    ) {
        self.texels[x as usize] = t.to_srgb_storage()
    }

    #[inline]
    fn clear(&mut self, t: <Channels<CHANNELS> as ChannelDesc>::Texel) {
        self.texels.fill([t.to_srgb_storage(); 4]);
    }
}

#[repr(transparent)]
pub struct MultiSampledColorTextureRowIter<'a, CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    chunks: std::slice::ChunksMut<
        'a,
        [<<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage; 4],
    >,
}

impl<'a, CS: ColorSpace, const CHANNELS: usize> Iterator
    for MultiSampledColorTextureRowIter<'a, CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Item = MultiSampledColorTextureRow<'a, CS, CHANNELS>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks
            .next()
            .map(|texels| MultiSampledColorTextureRow {
                _cs: PhantomData,
                texels,
            })
    }
}

#[repr(transparent)]
pub struct MultiSampledColorTextureParRowIter<'a, CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    chunks: rayon::slice::ChunksMut<
        'a,
        [<<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage; 4],
    >,
}

impl<'a, CS: ColorSpace, const CHANNELS: usize> ParallelIterator
    for MultiSampledColorTextureParRowIter<'a, CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Item = MultiSampledColorTextureRow<'a, CS, CHANNELS>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        self.chunks
            .map(|texels| MultiSampledColorTextureRow {
                _cs: PhantomData,
                texels,
            })
            .drive_unindexed(consumer)
    }
}

impl<'a, CS: ColorSpace, const CHANNELS: usize> IndexedParallelIterator
    for MultiSampledColorTextureParRowIter<'a, CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.chunks
            .map(|texels| MultiSampledColorTextureRow {
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
            .map(|texels| MultiSampledColorTextureRow {
                _cs: PhantomData,
                texels,
            })
            .with_producer(callback)
    }
}

pub struct MultiSampledColorTexture<CS: ColorSpace, const CHANNELS: usize>
where
    Channels<CHANNELS>: ChannelDesc,
{
    _cs: PhantomData<fn(CS)>,
    width: u32,
    height: u32,
    texels: Box<[[<<Channels<CHANNELS> as ChannelDesc>::Texel as Texel>::Storage; 4]]>,
}

impl<CS: ColorSpace, const CHANNELS: usize> MultiSampledColorTexture<CS, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    pub fn new(width: u32, height: u32) -> Self {
        let texel_size = (width as usize) * (height as usize);

        Self {
            _cs: PhantomData,
            width,
            height,
            texels: unsafe { Box::new_zeroed_slice(texel_size).assume_init() },
        }
    }
}

impl<const CHANNELS: usize> MultiSampledTexture for MultiSampledColorTexture<Linear, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Resolved = ColorTexture<Linear, CHANNELS>;
    type Row<'a> = MultiSampledColorTextureRow<'a, Linear, CHANNELS>;
    type RowIter<'a> = MultiSampledColorTextureRowIter<'a, Linear, CHANNELS>;
    type ParRowIter<'a> = MultiSampledColorTextureParRowIter<'a, Linear, CHANNELS>;

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn get_texel(
        &self,
        x: u32,
        y: u32,
    ) -> <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled::from_storage(
            self.texels[texel_index],
        )
    }

    #[inline]
    fn set_texel(
        &mut self,
        x: u32,
        y: u32,
        t: <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled,
    ) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[texel_index] = t.to_storage();
    }

    #[inline]
    fn rows_mut(&mut self) -> Self::RowIter<'_> {
        MultiSampledColorTextureRowIter {
            _cs: PhantomData,
            chunks: self.texels.chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_> {
        use rayon::prelude::*;
        MultiSampledColorTextureParRowIter {
            _cs: PhantomData,
            chunks: self.texels.par_chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn clear(&mut self, t: <Self::Resolved as Texture>::Texel) {
        self.texels.fill([t.to_storage(); 4]);
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;

        let texel_size = (new_width as usize) * (new_height as usize);
        self.texels = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
    }

    fn resolve_into(&mut self, target: &mut Self::Resolved) {
        assert_eq!(self.width(), target.width());
        assert_eq!(self.height(), target.height());

        self.par_rows_mut()
            .zip(target.par_rows_mut())
            .for_each(|(src_row, mut dst_row)| {
                for x in 0..src_row.width() {
                    let t = src_row.get_texel(x).resolve();
                    dst_row.set_texel(x, t);
                }
            });
    }
}

impl<const CHANNELS: usize> MultiSampledTexture for MultiSampledColorTexture<Srgb, CHANNELS>
where
    Channels<CHANNELS>: ChannelDesc,
{
    type Resolved = ColorTexture<Srgb, CHANNELS>;
    type Row<'a> = MultiSampledColorTextureRow<'a, Srgb, CHANNELS>;
    type RowIter<'a> = MultiSampledColorTextureRowIter<'a, Srgb, CHANNELS>;
    type ParRowIter<'a> = MultiSampledColorTextureParRowIter<'a, Srgb, CHANNELS>;

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn get_texel(
        &self,
        x: u32,
        y: u32,
    ) -> <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled::from_srgb_storage(
            self.texels[texel_index],
        )
    }

    #[inline]
    fn set_texel(
        &mut self,
        x: u32,
        y: u32,
        t: <<Self::Resolved as Texture>::Texel as Texel>::MultiSampled,
    ) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[texel_index] = t.to_srgb_storage();
    }

    #[inline]
    fn rows_mut(&mut self) -> Self::RowIter<'_> {
        MultiSampledColorTextureRowIter {
            _cs: PhantomData,
            chunks: self.texels.chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_> {
        use rayon::prelude::*;
        MultiSampledColorTextureParRowIter {
            _cs: PhantomData,
            chunks: self.texels.par_chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn clear(&mut self, t: <Self::Resolved as Texture>::Texel) {
        self.texels.fill([t.to_srgb_storage(); 4]);
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;

        let texel_size = (new_width as usize) * (new_height as usize);
        self.texels = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
    }

    fn resolve_into(&mut self, target: &mut Self::Resolved) {
        assert_eq!(self.width(), target.width());
        assert_eq!(self.height(), target.height());

        self.par_rows_mut()
            .zip(target.par_rows_mut())
            .for_each(|(src_row, mut dst_row)| {
                for x in 0..src_row.width() {
                    let t = src_row.get_texel(x).resolve();
                    dst_row.set_texel(x, t);
                }
            });
    }
}

#[repr(transparent)]
pub struct MultiSampledDepthTextureRow<'a> {
    texels: &'a mut [[f32; 4]],
}

impl<'a> MultiSampledTextureRow<'a, f32> for MultiSampledDepthTextureRow<'a> {
    #[inline]
    fn width(&self) -> u32 {
        self.texels.len() as u32
    }

    #[inline]
    fn get_texel(&self, x: u32) -> [f32; 4] {
        self.texels[x as usize]
    }

    #[inline]
    fn set_texel(&mut self, x: u32, t: [f32; 4]) {
        self.texels[x as usize] = t
    }

    #[inline]
    fn clear(&mut self, t: f32) {
        self.texels.fill([t; 4]);
    }
}

#[repr(transparent)]
pub struct MultiSampledDepthTextureRowIter<'a> {
    chunks: std::slice::ChunksMut<'a, [f32; 4]>,
}

impl<'a> Iterator for MultiSampledDepthTextureRowIter<'a> {
    type Item = MultiSampledDepthTextureRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks
            .next()
            .map(|texels| MultiSampledDepthTextureRow { texels })
    }
}

#[repr(transparent)]
pub struct MultiSampledDepthTextureParRowIter<'a> {
    chunks: rayon::slice::ChunksMut<'a, [f32; 4]>,
}

impl<'a> ParallelIterator for MultiSampledDepthTextureParRowIter<'a> {
    type Item = MultiSampledDepthTextureRow<'a>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        self.chunks
            .map(|texels| MultiSampledDepthTextureRow { texels })
            .drive_unindexed(consumer)
    }
}

impl<'a> IndexedParallelIterator for MultiSampledDepthTextureParRowIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.chunks
            .map(|texels| MultiSampledDepthTextureRow { texels })
            .drive(consumer)
    }

    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
        self,
        callback: CB,
    ) -> CB::Output {
        self.chunks
            .map(|texels| MultiSampledDepthTextureRow { texels })
            .with_producer(callback)
    }
}

pub struct MultiSampledDepthTexture {
    width: u32,
    height: u32,
    texels: Box<[[f32; 4]]>,
}

impl MultiSampledDepthTexture {
    pub fn new(width: u32, height: u32) -> Self {
        let texel_size = (width as usize) * (height as usize);

        Self {
            width,
            height,
            texels: unsafe { Box::new_zeroed_slice(texel_size).assume_init() },
        }
    }
}

impl MultiSampledTexture for MultiSampledDepthTexture {
    type Resolved = DepthTexture;
    type Row<'a> = MultiSampledDepthTextureRow<'a>;
    type RowIter<'a> = MultiSampledDepthTextureRowIter<'a>;
    type ParRowIter<'a> = MultiSampledDepthTextureParRowIter<'a>;

    #[inline]
    fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    fn get_texel(&self, x: u32, y: u32) -> [f32; 4] {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[texel_index]
    }

    #[inline]
    fn set_texel(&mut self, x: u32, y: u32, t: [f32; 4]) {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);

        let texel_index = (x as usize) + ((y as usize) * (self.width as usize));
        self.texels[texel_index] = t;
    }

    #[inline]
    fn rows_mut(&mut self) -> Self::RowIter<'_> {
        MultiSampledDepthTextureRowIter {
            chunks: self.texels.chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn par_rows_mut(&mut self) -> Self::ParRowIter<'_> {
        use rayon::prelude::*;
        MultiSampledDepthTextureParRowIter {
            chunks: self.texels.par_chunks_mut(self.width as usize),
        }
    }

    #[inline]
    fn clear(&mut self, t: f32) {
        self.texels.fill([t; 4]);
    }

    fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;

        let texel_size = (new_width as usize) * (new_height as usize);
        self.texels = unsafe { Box::new_zeroed_slice(texel_size).assume_init() };
    }

    fn resolve_into(&mut self, target: &mut Self::Resolved) {
        assert_eq!(self.width(), target.width());
        assert_eq!(self.height(), target.height());

        self.par_rows_mut()
            .zip(target.par_rows_mut())
            .for_each(|(src_row, mut dst_row)| {
                for x in 0..src_row.width() {
                    let t = src_row.get_texel(x).resolve();
                    dst_row.set_texel(x, t);
                }
            });
    }
}
