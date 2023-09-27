use crate::multi_sampled_texture::*;
use crate::texture::*;
use itertools::*;
use rayon::prelude::*;
use slender_math::*;
use smallvec::{smallvec, SmallVec};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

pub trait VertexOutput: Copy + Send + Sync {
    fn lerp(&self, rhs: &Self, t: f32) -> Self;
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, w: f32) -> Self;
}

impl VertexOutput for () {
    #[inline]
    fn lerp(&self, _rhs: &Self, _t: f32) -> Self {}

    #[inline]
    fn add(&self, _rhs: &Self) -> Self {}

    #[inline]
    fn sub(&self, _rhs: &Self) -> Self {}

    #[inline]
    fn mul(&self, _w: f32) -> Self {}
}

impl VertexOutput for f32 {
    #[inline]
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        *self + ((*rhs - *self) * t)
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    #[inline]
    fn mul(&self, w: f32) -> Self {
        *self * w
    }
}

impl VertexOutput for v2f {
    #[inline]
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        v2f::lerp(*self, *rhs, t)
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    #[inline]
    fn mul(&self, w: f32) -> Self {
        *self * w
    }
}

impl VertexOutput for v3f {
    #[inline]
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        v3f::lerp(*self, *rhs, t)
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    #[inline]
    fn mul(&self, w: f32) -> Self {
        *self * w
    }
}

impl VertexOutput for v4f {
    #[inline]
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        v4f::lerp(*self, *rhs, t)
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        *self + *rhs
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        *self - *rhs
    }

    #[inline]
    fn mul(&self, w: f32) -> Self {
        *self / w
    }
}

pub trait VertexShader {
    type Input: Copy + Send + Sync;
    type Output: VertexOutput;
    type Uniform: Sync;

    fn run(input: Self::Input, uniform: &Self::Uniform) -> (v4f, Self::Output);
}

pub trait PixelShader {
    type Input: VertexOutput;
    type Output: Texel;
    type Uniform: Sync;

    fn run(
        input: Self::Input,
        dx: Self::Input,
        dy: Self::Input,
        uniform: &Self::Uniform,
    ) -> Self::Output;
}

pub trait Index: Copy + Send + Sync {
    fn to_usize(self) -> usize;
}

impl Index for u8 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl Index for u16 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl Index for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

struct Vertex<VS: VertexShader> {
    pos: v4f,
    data: <VS as VertexShader>::Output,
}

impl<VS: VertexShader> Vertex<VS> {
    #[inline]
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        Self {
            pos: self.pos.lerp(rhs.pos, t),
            data: self.data.lerp(&rhs.data, t),
        }
    }
}

impl<VS: VertexShader> Add for Vertex<VS> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            pos: self.pos + rhs.pos,
            data: self.data.add(&rhs.data),
        }
    }
}

impl<VS: VertexShader> Sub for Vertex<VS> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            pos: self.pos - rhs.pos,
            data: self.data.sub(&rhs.data),
        }
    }
}

impl<VS: VertexShader> Mul<f32> for Vertex<VS> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            pos: self.pos * rhs,
            data: self.data.mul(rhs),
        }
    }
}

impl<VS: VertexShader> Div<f32> for Vertex<VS> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        let rhs_inv = rhs.recip();

        Self {
            pos: self.pos * rhs_inv,
            data: self.data.mul(rhs_inv),
        }
    }
}

impl<VS: VertexShader> From<(v4f, <VS as VertexShader>::Output)> for Vertex<VS> {
    #[inline]
    fn from(value: (v4f, <VS as VertexShader>::Output)) -> Self {
        Self {
            pos: value.0,
            data: value.1,
        }
    }
}

impl<VS: VertexShader> Clone for Vertex<VS> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            pos: self.pos,
            data: self.data,
        }
    }
}

impl<VS: VertexShader> Copy for Vertex<VS> {}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontFace {
    None,
    Ccw,
    Cw,
}

struct Triangle<VS: VertexShader> {
    a: Vertex<VS>,
    b: Vertex<VS>,
    c: Vertex<VS>,
}

impl<VS: VertexShader> Triangle<VS> {
    #[inline]
    const fn new(a: Vertex<VS>, b: Vertex<VS>, c: Vertex<VS>) -> Self {
        Self { a, b, c }
    }

    #[inline]
    fn min_x(&self) -> f32 {
        self.a.pos.x().min(self.b.pos.x()).min(self.c.pos.x())
    }

    #[inline]
    fn max_x(&self) -> f32 {
        self.a.pos.x().max(self.b.pos.x()).max(self.c.pos.x())
    }

    fn is_in_frustum(&self) -> bool {
        if (self.a.pos.x() > self.a.pos.w())
            && (self.b.pos.x() > self.b.pos.w())
            && (self.c.pos.x() > self.c.pos.w())
        {
            return false;
        }

        if (self.a.pos.x() < -self.a.pos.w())
            && (self.b.pos.x() < -self.b.pos.w())
            && (self.c.pos.x() < -self.c.pos.w())
        {
            return false;
        }

        if (self.a.pos.y() > self.a.pos.w())
            && (self.b.pos.y() > self.b.pos.w())
            && (self.c.pos.y() > self.c.pos.w())
        {
            return false;
        }

        if (self.a.pos.y() < -self.a.pos.w())
            && (self.b.pos.y() < -self.b.pos.w())
            && (self.c.pos.y() < -self.c.pos.w())
        {
            return false;
        }

        if (self.a.pos.z() > self.a.pos.w())
            && (self.b.pos.z() > self.b.pos.w())
            && (self.c.pos.z() > self.c.pos.w())
        {
            return false;
        }

        if (self.a.pos.z() < 0.0) && (self.b.pos.z() < 0.0) && (self.c.pos.z() < 0.0) {
            return false;
        }

        true
    }

    fn transform_to_screen_space(&mut self, scaling: v4f, translation: v4f) {
        const BASE_TRANSLATION: v4f = v4f::new(1.0, 1.0, 0.0, 0.0);

        let aw_inv = self.a.pos.w().recip();
        let bw_inv = self.b.pos.w().recip();
        let cw_inv = self.c.pos.w().recip();

        self.a.pos = (self.a.pos * aw_inv + BASE_TRANSLATION) * scaling + translation;
        self.b.pos = (self.b.pos * bw_inv + BASE_TRANSLATION) * scaling + translation;
        self.c.pos = (self.c.pos * cw_inv + BASE_TRANSLATION) * scaling + translation;

        *self.a.pos.w_mut() = aw_inv;
        *self.b.pos.w_mut() = bw_inv;
        *self.c.pos.w_mut() = cw_inv;

        self.a.data = self.a.data.mul(aw_inv);
        self.b.data = self.b.data.mul(bw_inv);
        self.c.data = self.c.data.mul(cw_inv);
    }

    fn is_front_facing(&self, front_face: FrontFace) -> bool {
        if front_face == FrontFace::None {
            return true;
        }

        let ab = self.b.pos.xy() - self.a.pos.xy();
        let ac = self.c.pos.xy() - self.a.pos.xy();
        let dp = ab.cross(ac);
        match front_face {
            FrontFace::None => true,
            FrontFace::Ccw => dp < 0.0,
            FrontFace::Cw => dp > 0.0,
        }
    }

    fn contains_point_2d(&self, p: v2f) -> bool {
        let a = self.a.pos.xy();
        let b = self.b.pos.xy();
        let c = self.c.pos.xy();

        let ca = a - c;
        let ab = b - a;
        let cp = p - c;
        let ap = p - a;
        let s = ca.cross(cp);
        let t = ab.cross(ap);

        if ((s < 0.0) != (t < 0.0)) && (s != 0.0) && (t != 0.0) {
            return false;
        }

        let bc = c - b;
        let bp = p - c;
        let d = bc.cross(bp);

        (d == 0.0) || ((d < 0.0) == (s + t <= 0.0))
    }
}

enum FlatTriangle<VS: VertexShader> {
    FlatBottom(Triangle<VS>),
    FlatTop(Triangle<VS>),
}

impl<VS: VertexShader> FlatTriangle<VS> {
    #[inline]
    const fn new_bottom(a: Vertex<VS>, b: Vertex<VS>, c: Vertex<VS>) -> Self {
        Self::FlatBottom(Triangle::new(a, b, c))
    }

    #[inline]
    const fn new_top(a: Vertex<VS>, b: Vertex<VS>, c: Vertex<VS>) -> Self {
        Self::FlatTop(Triangle::new(a, b, c))
    }

    #[inline]
    fn start_end_y(&self, max_y: usize) -> (usize, usize) {
        match self {
            FlatTriangle::FlatBottom(tri) | FlatTriangle::FlatTop(tri) => {
                let start_y = ((tri.a.pos.y() - 0.5).ceil() as usize).clamp(0, max_y);
                let end_y = ((tri.c.pos.y() - 0.5).ceil() as usize).clamp(0, max_y);

                (start_y, end_y)
            }
        }
    }

    #[inline]
    fn start_end_y_multi_sampled(&self, max_y: usize) -> (usize, usize) {
        match self {
            FlatTriangle::FlatBottom(tri) | FlatTriangle::FlatTop(tri) => {
                let start_y = ((tri.a.pos.y() - 0.5).floor() as usize).clamp(0, max_y);
                let end_y = ((tri.c.pos.y() + 1.5).ceil() as usize).clamp(0, max_y);

                (start_y, end_y)
            }
        }
    }
}

fn clip_triangle<VS: VertexShader>(tri: Triangle<VS>) -> SmallVec<[Triangle<VS>; 2]> {
    use std::cmp::Ordering::Less;

    match (
        tri.a.pos.z().total_cmp(&0.0),
        tri.b.pos.z().total_cmp(&0.0),
        tri.c.pos.z().total_cmp(&0.0),
    ) {
        (Less, Less, _) => {
            let t0 = tri.a.pos.z() / (tri.a.pos.z() - tri.c.pos.z());
            let t1 = tri.b.pos.z() / (tri.b.pos.z() - tri.c.pos.z());

            let v0 = tri.a.lerp(&tri.c, t0);
            let v1 = tri.b.lerp(&tri.c, t1);

            smallvec![Triangle::new(v0, v1, tri.c)]
        }
        (Less, _, Less) => {
            let t0 = tri.a.pos.z() / (tri.a.pos.z() - tri.b.pos.z());
            let t1 = tri.c.pos.z() / (tri.c.pos.z() - tri.b.pos.z());

            let v0 = tri.a.lerp(&tri.b, t0);
            let v1 = tri.c.lerp(&tri.b, t1);

            smallvec![Triangle::new(v0, tri.b, v1)]
        }
        (_, Less, Less) => {
            let t0 = tri.b.pos.z() / (tri.b.pos.z() - tri.a.pos.z());
            let t1 = tri.c.pos.z() / (tri.c.pos.z() - tri.a.pos.z());

            let v0 = tri.b.lerp(&tri.a, t0);
            let v1 = tri.c.lerp(&tri.a, t1);

            smallvec![Triangle::new(tri.a, v0, v1)]
        }
        (Less, _, _) => {
            let t0 = tri.a.pos.z() / (tri.a.pos.z() - tri.b.pos.z());
            let t1 = tri.a.pos.z() / (tri.a.pos.z() - tri.c.pos.z());

            let v0 = tri.a.lerp(&tri.b, t0);
            let v1 = tri.a.lerp(&tri.c, t1);

            smallvec![
                Triangle::new(v0, tri.b, tri.c),
                Triangle::new(v1, v0, tri.c),
            ]
        }
        (_, Less, _) => {
            let t0 = tri.b.pos.z() / (tri.b.pos.z() - tri.a.pos.z());
            let t1 = tri.b.pos.z() / (tri.b.pos.z() - tri.c.pos.z());

            let v0 = tri.b.lerp(&tri.a, t0);
            let v1 = tri.b.lerp(&tri.c, t1);

            smallvec![
                Triangle::new(tri.a, v0, tri.c),
                Triangle::new(v0, v1, tri.c),
            ]
        }
        (_, _, Less) => {
            let t0 = tri.c.pos.z() / (tri.c.pos.z() - tri.a.pos.z());
            let t1 = tri.c.pos.z() / (tri.c.pos.z() - tri.b.pos.z());

            let v0 = tri.c.lerp(&tri.a, t0);
            let v1 = tri.c.lerp(&tri.b, t1);

            smallvec![
                Triangle::new(tri.a, tri.b, v0),
                Triangle::new(v0, tri.b, v1),
            ]
        }
        (_, _, _) => smallvec![tri],
    }
}

fn process_triangle<VS: VertexShader>(tri: Triangle<VS>) -> SmallVec<[FlatTriangle<VS>; 2]> {
    let mut a = &tri.a;
    let mut b = &tri.b;
    let mut c = &tri.c;

    if b.pos.y() < a.pos.y() {
        std::mem::swap(&mut a, &mut b);
    }
    if c.pos.y() < b.pos.y() {
        std::mem::swap(&mut b, &mut c);
    }
    if b.pos.y() < a.pos.y() {
        std::mem::swap(&mut a, &mut b);
    }

    if a.pos.y() == b.pos.y() {
        if b.pos.x() < a.pos.x() {
            std::mem::swap(&mut a, &mut b);
        }

        smallvec![FlatTriangle::new_top(*a, *b, *c)]
    } else if b.pos.y() == c.pos.y() {
        if c.pos.x() < b.pos.x() {
            std::mem::swap(&mut b, &mut c);
        }

        smallvec![FlatTriangle::new_bottom(*a, *b, *c)]
    } else {
        let t = (b.pos.y() - a.pos.y()) / (c.pos.y() - a.pos.y());
        let v = a.lerp(&c, t);

        if b.pos.x() < v.pos.x() {
            smallvec![
                FlatTriangle::new_bottom(*a, *b, v),
                FlatTriangle::new_top(*b, v, *c),
            ]
        } else {
            smallvec![
                FlatTriangle::new_bottom(*a, v, *b),
                FlatTriangle::new_top(v, *b, *c),
            ]
        }
    }
}

pub struct Pipeline<VS, PS>
where
    VS: VertexShader,
    PS: PixelShader<Input = <VS as VertexShader>::Output, Uniform = <VS as VertexShader>::Uniform>,
{
    _vs: PhantomData<fn(VS::Input) -> VS::Output>,
    _ps: PhantomData<fn(PS::Input) -> PS::Output>,
    chunk_count: usize,
    blend_mode: BlendMode,
    front_face: FrontFace,
    triangle_lookup: Box<[Vec<u32>]>,
}

impl<VS, PS> Pipeline<VS, PS>
where
    VS: VertexShader,
    PS: PixelShader<Input = <VS as VertexShader>::Output, Uniform = <VS as VertexShader>::Uniform>,
{
    pub fn new(blend_mode: BlendMode, front_face: FrontFace) -> Self {
        let num_threads = rayon::current_num_threads();
        let chunk_count = num_threads * 24;

        Self {
            _vs: PhantomData,
            _ps: PhantomData,
            chunk_count,
            blend_mode,
            front_face,
            triangle_lookup: vec![Vec::new(); chunk_count].into_boxed_slice(),
        }
    }

    fn draw_triangle_row<T: Texture<Texel = PS::Output>>(
        &self,
        color_row: &mut T::Row<'_>,
        mut depth_row: Option<&mut <DepthTexture as Texture>::Row<'_>>,
        v0: Vertex<VS>,
        v1: Vertex<VS>,
        v0_sub: Vertex<VS>,
        v1_sub: Vertex<VS>,
        min_x: u32,
        max_x: u32,
        uniform: &PS::Uniform,
    ) {
        let start_x = ((v0.pos.x() - 0.5).ceil() as u32).clamp(min_x, max_x);
        let end_x = ((v1.pos.x() - 0.5).ceil() as u32).clamp(min_x, max_x);
        if start_x >= end_x {
            return;
        }

        let diff_x = v1.pos.x() - v0.pos.x();
        let start_t = ((start_x as f32) - v0.pos.x()) / diff_x;
        let end_t = ((end_x as f32) - v0.pos.x()) / diff_x;
        let start_v = v0.lerp(&v1, start_t);
        let end_v = v0.lerp(&v1, end_t);
        let step_v = (end_v - start_v) / ((end_x - start_x) as f32);

        let diff_x_sub = v1_sub.pos.x() - v0_sub.pos.x();
        let start_t_sub = ((start_x as f32) - v0_sub.pos.x()) / diff_x_sub;
        let end_t_sub = ((end_x as f32) - v0_sub.pos.x()) / diff_x_sub;
        let start_v_sub = v0_sub.lerp(&v1_sub, start_t_sub);
        let end_v_sub = v0_sub.lerp(&v1_sub, end_t_sub);
        let step_v_sub = (end_v_sub - start_v_sub) / ((end_x - start_x) as f32);

        let mut current_v = start_v;
        let mut current_data = current_v.data.mul(current_v.pos.w().recip());
        let mut current_v_sub = start_v_sub;
        for x in start_x..end_x {
            let next_v = current_v + step_v;
            let next_data = next_v.data.mul(next_v.pos.w().recip());

            if depth_row
                .as_ref()
                .map_or(true, |depth_row| depth_row.get_texel(x) > current_v.pos.z())
            {
                let current_data_sub = current_v_sub.data.mul(current_v_sub.pos.w().recip());

                let dx = next_data.sub(&current_data);
                let dy = current_data_sub.sub(&current_data);

                let ps_output = PS::run(current_data, dx, dy, uniform).clamp_to_storage_range();

                if color_row.blend_texel(x, ps_output, self.blend_mode) {
                    if let Some(depth_row) = depth_row.as_mut() {
                        depth_row.set_texel(x, current_v.pos.z());
                    }
                }
            }

            current_v = next_v;
            current_data = next_data;
            current_v_sub = current_v_sub + step_v_sub;
        }
    }

    fn draw_flat_bottom_triangle_row<T: Texture<Texel = PS::Output>>(
        &self,
        y: usize,
        color_row: &mut T::Row<'_>,
        depth_row: Option<&mut <DepthTexture as Texture>::Row<'_>>,
        tri: &Triangle<VS>,
        uniform: &PS::Uniform,
    ) {
        let yf = y as f32;
        let diff_y = tri.c.pos.y() - tri.a.pos.y();
        let ty_0 = (yf - tri.a.pos.y()) / diff_y;
        let ty_1 = (yf - tri.a.pos.y() + 1.0) / diff_y;
        let v0 = tri.a.lerp(&tri.b, ty_0);
        let v1 = tri.a.lerp(&tri.c, ty_0);
        let v0_sub = tri.a.lerp(&tri.b, ty_1);
        let v1_sub = tri.a.lerp(&tri.c, ty_1);

        let min_x = (tri.min_x().floor().max(0.0) as u32).min(color_row.width());
        let max_x = (tri.max_x().ceil().max(0.0) as u32).min(color_row.width());

        self.draw_triangle_row::<T>(
            color_row, depth_row, v0, v1, v0_sub, v1_sub, min_x, max_x, uniform,
        );
    }

    fn draw_flat_top_triangle_row<T: Texture<Texel = PS::Output>>(
        &self,
        y: usize,
        color_row: &mut T::Row<'_>,
        depth_row: Option<&mut <DepthTexture as Texture>::Row<'_>>,
        tri: &Triangle<VS>,
        uniform: &PS::Uniform,
    ) {
        let yf = y as f32;
        let diff_y = tri.c.pos.y() - tri.a.pos.y();
        let ty_0 = (yf - tri.a.pos.y()) / diff_y;
        let ty_1 = (yf - tri.a.pos.y() + 1.0) / diff_y;
        let v0 = tri.a.lerp(&tri.c, ty_0);
        let v1 = tri.b.lerp(&tri.c, ty_0);
        let v0_sub = tri.a.lerp(&tri.c, ty_1);
        let v1_sub = tri.b.lerp(&tri.c, ty_1);

        let min_x = (tri.min_x().floor().max(0.0) as u32).min(color_row.width());
        let max_x = (tri.max_x().ceil().max(0.0) as u32).min(color_row.width());

        self.draw_triangle_row::<T>(
            color_row, depth_row, v0, v1, v0_sub, v1_sub, min_x, max_x, uniform,
        );
    }

    fn draw_triangles<T: Texture<Texel = PS::Output>>(
        &self,
        color_buffer: &mut T,
        depth_buffer: &mut DepthTexture,
        uniform: &VS::Uniform,
        chunk_size: usize,
        buffer_height: usize,
        triangles: &[FlatTriangle<VS>],
    ) {
        let mut rows: Vec<_> = izip!(color_buffer.rows_mut(), depth_buffer.rows_mut()).collect();
        rows.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, rows)| {
                let chunk_start_y = chunk_index * chunk_size;
                let chunk_end_y = (chunk_start_y + chunk_size).min(buffer_height);

                for tri in self.triangle_lookup[chunk_index]
                    .iter()
                    .map(|&tri_index| &triangles[tri_index as usize])
                {
                    let (tri_start_y, tri_end_y) = tri.start_end_y(buffer_height);
                    let start_y = tri_start_y.clamp(chunk_start_y, chunk_end_y);
                    let end_y = tri_end_y.clamp(chunk_start_y, chunk_end_y);

                    let row_offset = start_y - chunk_start_y;
                    let row_count = end_y - chunk_start_y - row_offset;

                    for (offset_y, (ref mut color_row, ref mut depth_row)) in
                        rows.iter_mut().enumerate().skip(row_offset).take(row_count)
                    {
                        match tri {
                            FlatTriangle::FlatBottom(tri) => {
                                self.draw_flat_bottom_triangle_row::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    Some(depth_row),
                                    tri,
                                    uniform,
                                );
                            }
                            FlatTriangle::FlatTop(tri) => {
                                self.draw_flat_top_triangle_row::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    Some(depth_row),
                                    tri,
                                    uniform,
                                );
                            }
                        }
                    }
                }
            });
    }

    fn draw_triangles_no_depth<T: Texture<Texel = PS::Output>>(
        &self,
        color_buffer: &mut T,
        uniform: &VS::Uniform,
        chunk_size: usize,
        buffer_height: usize,
        triangles: &[FlatTriangle<VS>],
    ) {
        let mut rows: Vec<_> = color_buffer.rows_mut().collect();
        rows.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, rows)| {
                let chunk_start_y = chunk_index * chunk_size;
                let chunk_end_y = (chunk_start_y + chunk_size).min(buffer_height);

                for tri in self.triangle_lookup[chunk_index]
                    .iter()
                    .map(|&tri_index| &triangles[tri_index as usize])
                {
                    let (tri_start_y, tri_end_y) = tri.start_end_y(buffer_height);
                    let start_y = tri_start_y.clamp(chunk_start_y, chunk_end_y);
                    let end_y = tri_end_y.clamp(chunk_start_y, chunk_end_y);

                    let row_offset = start_y - chunk_start_y;
                    let row_count = end_y - chunk_start_y - row_offset;

                    for (offset_y, ref mut color_row) in
                        rows.iter_mut().enumerate().skip(row_offset).take(row_count)
                    {
                        match tri {
                            FlatTriangle::FlatBottom(tri) => {
                                self.draw_flat_bottom_triangle_row::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    None,
                                    tri,
                                    uniform,
                                );
                            }
                            FlatTriangle::FlatTop(tri) => {
                                self.draw_flat_top_triangle_row::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    None,
                                    tri,
                                    uniform,
                                );
                            }
                        }
                    }
                }
            });
    }

    pub fn draw_indexed<T: Texture<Texel = PS::Output>, I: Index>(
        &mut self,
        color_buffer: &mut T,
        depth_buffer: Option<&mut DepthTexture>,
        vertices: &[VS::Input],
        indices: &[I],
        uniform: &VS::Uniform,
    ) {
        if let Some(depth_buffer) = depth_buffer.as_ref() {
            assert_eq!(color_buffer.width(), depth_buffer.width());
            assert_eq!(color_buffer.height(), depth_buffer.height());
        }
        assert!(vertices.len() > 2);
        assert_eq!(indices.len() % 3, 0);

        let transformed_vertices: Vec<Vertex<VS>> = vertices
            .par_iter()
            .copied()
            .map(|vertex| VS::run(vertex, uniform).into())
            .collect();

        let buffer_width_f = color_buffer.width() as f32;
        let buffer_height_f = color_buffer.height() as f32;
        let scaling = v4f::new(0.5 * buffer_width_f, -0.5 * buffer_height_f, 1.0, 1.0);
        let translation = v4f::new(0.0, buffer_height_f, 0.0, 0.0);

        let triangles: Vec<FlatTriangle<VS>> = (0..(indices.len() / 3))
            .into_par_iter()
            .flat_map_iter(|i| {
                let tri = Triangle::<VS>::new(
                    transformed_vertices[indices[i * 3 + 0].to_usize()],
                    transformed_vertices[indices[i * 3 + 1].to_usize()],
                    transformed_vertices[indices[i * 3 + 2].to_usize()],
                );

                let tri = if tri.is_in_frustum() { Some(tri) } else { None };

                tri.into_iter()
                    .flat_map(clip_triangle)
                    .update(|tri| tri.transform_to_screen_space(scaling, translation))
                    .filter(|tri| tri.is_front_facing(self.front_face))
                    .flat_map(process_triangle)
            })
            .collect();

        let buffer_height = color_buffer.height() as usize;
        let chunk_size =
            (((buffer_height as f32) / (self.chunk_count as f32)).ceil() as usize).max(1);

        for lookup in self.triangle_lookup.iter_mut() {
            lookup.clear();
        }

        for (tri_index, tri) in triangles.iter().enumerate() {
            let (start_y, end_y) = tri.start_end_y(buffer_height);

            if start_y < end_y {
                let start_chunk = start_y / chunk_size;
                let end_chunk = (end_y - 1) / chunk_size;

                for i in start_chunk..=end_chunk {
                    self.triangle_lookup[i].push(tri_index as u32);
                }
            }
        }

        if let Some(depth_buffer) = depth_buffer {
            self.draw_triangles(
                color_buffer,
                depth_buffer,
                uniform,
                chunk_size,
                buffer_height,
                &triangles,
            );
        } else {
            self.draw_triangles_no_depth(
                color_buffer,
                uniform,
                chunk_size,
                buffer_height,
                &triangles,
            );
        }
    }

    fn draw_triangle_row_multi_sampled<T: MultiSampledTexture>(
        &self,
        color_row: &mut T::Row<'_>,
        mut depth_row: Option<&mut <MultiSampledDepthTexture as MultiSampledTexture>::Row<'_>>,
        tri: &Triangle<VS>,
        v0: Vertex<VS>,
        v1: Vertex<VS>,
        v0_sub: Vertex<VS>,
        v1_sub: Vertex<VS>,
        min_x: u32,
        max_x: u32,
        uniform: &PS::Uniform,
    ) where
        T::Resolved: Texture<Texel = PS::Output>,
    {
        let start_x = ((v0.pos.x() - 0.5).floor() as u32).clamp(min_x, max_x);
        let end_x = ((v1.pos.x() + 1.5).ceil() as u32).clamp(min_x, max_x);
        if start_x >= end_x {
            return;
        }

        let diff_x = v1.pos.x() - v0.pos.x();
        let start_t = ((start_x as f32) - v0.pos.x()) / diff_x;
        let end_t = ((end_x as f32) - v0.pos.x()) / diff_x;
        let start_v = v0.lerp(&v1, start_t);
        let end_v = v0.lerp(&v1, end_t);
        let step_v = (end_v - start_v) / ((end_x - start_x) as f32);

        let diff_x_sub = v1_sub.pos.x() - v0_sub.pos.x();
        let start_t_sub = ((start_x as f32) - v0_sub.pos.x()) / diff_x_sub;
        let end_t_sub = ((end_x as f32) - v0_sub.pos.x()) / diff_x_sub;
        let start_v_sub = v0_sub.lerp(&v1_sub, start_t_sub);
        let end_v_sub = v0_sub.lerp(&v1_sub, end_t_sub);
        let step_v_sub = (end_v_sub - start_v_sub) / ((end_x - start_x) as f32);

        let mut current_v = start_v;
        let mut current_data = current_v.data.mul(current_v.pos.w().recip());
        let mut current_v_sub = start_v_sub;
        for x in start_x..end_x {
            let next_v = current_v + step_v;
            let next_data = next_v.data.mul(next_v.pos.w().recip());

            let current_data_sub = current_v_sub.data.mul(current_v_sub.pos.w().recip());

            let dx = next_data.sub(&current_data);
            let dy = current_data_sub.sub(&current_data);

            let ps_output = PS::run(current_data, dx, dy, uniform).clamp_to_storage_range();

            let p00 = current_v.pos.xy() + v2f::new(-0.25, -0.75);
            let p01 = current_v.pos.xy() + v2f::new(-0.75, 0.25);
            let p10 = current_v.pos.xy() + v2f::new(0.75, -0.25);
            let p11 = current_v.pos.xy() + v2f::new(0.25, 0.75);

            if let Some(depth_row) = depth_row.as_mut() {
                let mut d = depth_row.get_texel(x);
                let mut t = color_row.get_texel(x);
                if tri.contains_point_2d(p00) && (d.x0y0() > current_v.pos.z()) {
                    if let Some(tt) = ps_output.blend(t.x0y0(), self.blend_mode) {
                        d.set_x0y0(current_v.pos.z());
                        t.set_x0y0(tt);
                    }
                }
                if tri.contains_point_2d(p01) && (d.x0y1() > current_v.pos.z()) {
                    if let Some(tt) = ps_output.blend(t.x0y1(), self.blend_mode) {
                        d.set_x0y1(current_v.pos.z());
                        t.set_x0y1(tt);
                    }
                }
                if tri.contains_point_2d(p10) && (d.x1y0() > current_v.pos.z()) {
                    if let Some(tt) = ps_output.blend(t.x1y0(), self.blend_mode) {
                        d.set_x1y0(current_v.pos.z());
                        t.set_x1y0(tt);
                    }
                }
                if tri.contains_point_2d(p11) && (d.x1y1() > current_v.pos.z()) {
                    if let Some(tt) = ps_output.blend(t.x1y1(), self.blend_mode) {
                        d.set_x1y1(current_v.pos.z());
                        t.set_x1y1(tt);
                    }
                }
                depth_row.set_texel(x, d);
                color_row.set_texel(x, t);
            } else {
                let mut t = color_row.get_texel(x);
                if tri.contains_point_2d(p00) {
                    if let Some(tt) = ps_output.blend(t.x0y0(), self.blend_mode) {
                        t.set_x0y0(tt);
                    }
                }
                if tri.contains_point_2d(p01) {
                    if let Some(tt) = ps_output.blend(t.x0y1(), self.blend_mode) {
                        t.set_x0y1(tt);
                    }
                }
                if tri.contains_point_2d(p10) {
                    if let Some(tt) = ps_output.blend(t.x1y0(), self.blend_mode) {
                        t.set_x1y0(tt);
                    }
                }
                if tri.contains_point_2d(p11) {
                    if let Some(tt) = ps_output.blend(t.x1y1(), self.blend_mode) {
                        t.set_x1y1(tt);
                    }
                }
                color_row.set_texel(x, t);
            }

            current_v = next_v;
            current_data = next_data;
            current_v_sub = current_v_sub + step_v_sub;
        }
    }

    fn draw_flat_bottom_triangle_row_multi_sampled<T: MultiSampledTexture>(
        &self,
        y: usize,
        color_row: &mut T::Row<'_>,
        depth_row: Option<&mut <MultiSampledDepthTexture as MultiSampledTexture>::Row<'_>>,
        tri: &Triangle<VS>,
        uniform: &PS::Uniform,
    ) where
        T::Resolved: Texture<Texel = PS::Output>,
    {
        let yf = y as f32;
        let diff_y = tri.c.pos.y() - tri.a.pos.y();
        let ty_0 = (yf - tri.a.pos.y()) / diff_y;
        let ty_1 = (yf - tri.a.pos.y() + 1.0) / diff_y;
        let v0 = tri.a.lerp(&tri.b, ty_0);
        let v1 = tri.a.lerp(&tri.c, ty_0);
        let v0_sub = tri.a.lerp(&tri.b, ty_1);
        let v1_sub = tri.a.lerp(&tri.c, ty_1);

        let min_x = ((tri.min_x() - 0.5).floor().max(0.0) as u32).min(color_row.width());
        let max_x = ((tri.max_x() + 1.5).ceil().max(0.0) as u32).min(color_row.width());

        self.draw_triangle_row_multi_sampled::<T>(
            color_row, depth_row, tri, v0, v1, v0_sub, v1_sub, min_x, max_x, uniform,
        );
    }

    fn draw_flat_top_triangle_row_multi_sampled<T: MultiSampledTexture>(
        &self,
        y: usize,
        color_row: &mut T::Row<'_>,
        depth_row: Option<&mut <MultiSampledDepthTexture as MultiSampledTexture>::Row<'_>>,
        tri: &Triangle<VS>,
        uniform: &PS::Uniform,
    ) where
        T::Resolved: Texture<Texel = PS::Output>,
    {
        let yf = y as f32;
        let diff_y = tri.c.pos.y() - tri.a.pos.y();
        let ty_0 = (yf - tri.a.pos.y()) / diff_y;
        let ty_1 = (yf - tri.a.pos.y() + 1.0) / diff_y;
        let v0 = tri.a.lerp(&tri.c, ty_0);
        let v1 = tri.b.lerp(&tri.c, ty_0);
        let v0_sub = tri.a.lerp(&tri.c, ty_1);
        let v1_sub = tri.b.lerp(&tri.c, ty_1);

        let min_x = ((tri.min_x() - 0.5).floor().max(0.0) as u32).min(color_row.width());
        let max_x = ((tri.max_x() + 1.5).ceil().max(0.0) as u32).min(color_row.width());

        self.draw_triangle_row_multi_sampled::<T>(
            color_row, depth_row, tri, v0, v1, v0_sub, v1_sub, min_x, max_x, uniform,
        );
    }

    fn draw_triangles_multi_sampled<T: MultiSampledTexture>(
        &self,
        color_buffer: &mut T,
        depth_buffer: &mut MultiSampledDepthTexture,
        uniform: &VS::Uniform,
        chunk_size: usize,
        buffer_height: usize,
        triangles: &[FlatTriangle<VS>],
    ) where
        T::Resolved: Texture<Texel = PS::Output>,
    {
        let mut rows: Vec<_> = izip!(color_buffer.rows_mut(), depth_buffer.rows_mut()).collect();
        rows.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, rows)| {
                let chunk_start_y = chunk_index * chunk_size;
                let chunk_end_y = (chunk_start_y + chunk_size).min(buffer_height);

                for tri in self.triangle_lookup[chunk_index]
                    .iter()
                    .map(|&tri_index| &triangles[tri_index as usize])
                {
                    let (tri_start_y, tri_end_y) = tri.start_end_y_multi_sampled(buffer_height);
                    let start_y = tri_start_y.clamp(chunk_start_y, chunk_end_y);
                    let end_y = tri_end_y.clamp(chunk_start_y, chunk_end_y);

                    let row_offset = start_y - chunk_start_y;
                    let row_count = end_y - chunk_start_y - row_offset;

                    for (offset_y, (ref mut color_row, ref mut depth_row)) in
                        rows.iter_mut().enumerate().skip(row_offset).take(row_count)
                    {
                        match tri {
                            FlatTriangle::FlatBottom(tri) => {
                                self.draw_flat_bottom_triangle_row_multi_sampled::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    Some(depth_row),
                                    tri,
                                    uniform,
                                );
                            }
                            FlatTriangle::FlatTop(tri) => {
                                self.draw_flat_top_triangle_row_multi_sampled::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    Some(depth_row),
                                    tri,
                                    uniform,
                                );
                            }
                        }
                    }
                }
            });
    }

    fn draw_triangles_no_depth_multi_sampled<T: MultiSampledTexture>(
        &self,
        color_buffer: &mut T,
        uniform: &VS::Uniform,
        chunk_size: usize,
        buffer_height: usize,
        triangles: &[FlatTriangle<VS>],
    ) where
        T::Resolved: Texture<Texel = PS::Output>,
    {
        let mut rows: Vec<_> = color_buffer.rows_mut().collect();
        rows.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, rows)| {
                let chunk_start_y = chunk_index * chunk_size;
                let chunk_end_y = (chunk_start_y + chunk_size).min(buffer_height);

                for tri in self.triangle_lookup[chunk_index]
                    .iter()
                    .map(|&tri_index| &triangles[tri_index as usize])
                {
                    let (tri_start_y, tri_end_y) = tri.start_end_y_multi_sampled(buffer_height);
                    let start_y = tri_start_y.clamp(chunk_start_y, chunk_end_y);
                    let end_y = tri_end_y.clamp(chunk_start_y, chunk_end_y);

                    let row_offset = start_y - chunk_start_y;
                    let row_count = end_y - chunk_start_y - row_offset;

                    for (offset_y, ref mut color_row) in
                        rows.iter_mut().enumerate().skip(row_offset).take(row_count)
                    {
                        match tri {
                            FlatTriangle::FlatBottom(tri) => {
                                self.draw_flat_bottom_triangle_row_multi_sampled::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    None,
                                    tri,
                                    uniform,
                                );
                            }
                            FlatTriangle::FlatTop(tri) => {
                                self.draw_flat_top_triangle_row_multi_sampled::<T>(
                                    chunk_start_y + offset_y,
                                    color_row,
                                    None,
                                    tri,
                                    uniform,
                                );
                            }
                        }
                    }
                }
            });
    }

    pub fn draw_indexed_multi_sampled<T: MultiSampledTexture, I: Index>(
        &mut self,
        color_buffer: &mut T,
        depth_buffer: Option<&mut MultiSampledDepthTexture>,
        vertices: &[VS::Input],
        indices: &[I],
        uniform: &VS::Uniform,
    ) where
        T::Resolved: Texture<Texel = PS::Output>,
    {
        if let Some(depth_buffer) = depth_buffer.as_ref() {
            assert_eq!(color_buffer.width(), depth_buffer.width());
            assert_eq!(color_buffer.height(), depth_buffer.height());
        }
        assert!(vertices.len() > 2);
        assert_eq!(indices.len() % 3, 0);

        let transformed_vertices: Vec<Vertex<VS>> = vertices
            .par_iter()
            .copied()
            .map(|vertex| VS::run(vertex, uniform).into())
            .collect();

        let buffer_width_f = color_buffer.width() as f32;
        let buffer_height_f = color_buffer.height() as f32;
        let scaling = v4f::new(0.5 * buffer_width_f, -0.5 * buffer_height_f, 1.0, 1.0);
        let translation = v4f::new(0.0, buffer_height_f, 0.0, 0.0);

        let triangles: Vec<FlatTriangle<VS>> = (0..(indices.len() / 3))
            .into_par_iter()
            .flat_map_iter(|i| {
                let tri = Triangle::<VS>::new(
                    transformed_vertices[indices[i * 3 + 0].to_usize()],
                    transformed_vertices[indices[i * 3 + 1].to_usize()],
                    transformed_vertices[indices[i * 3 + 2].to_usize()],
                );

                let tri = if tri.is_in_frustum() { Some(tri) } else { None };

                tri.into_iter()
                    .flat_map(clip_triangle)
                    .update(|tri| tri.transform_to_screen_space(scaling, translation))
                    .filter(|tri| tri.is_front_facing(self.front_face))
                    .flat_map(process_triangle)
            })
            .collect();

        let buffer_height = color_buffer.height() as usize;
        let chunk_size =
            (((buffer_height as f32) / (self.chunk_count as f32)).ceil() as usize).max(1);

        for lookup in self.triangle_lookup.iter_mut() {
            lookup.clear();
        }

        for (tri_index, tri) in triangles.iter().enumerate() {
            let (start_y, end_y) = tri.start_end_y_multi_sampled(buffer_height);

            if start_y < end_y {
                let start_chunk = start_y / chunk_size;
                let end_chunk = (end_y - 1) / chunk_size;

                for i in start_chunk..=end_chunk {
                    self.triangle_lookup[i].push(tri_index as u32);
                }
            }
        }

        if let Some(depth_buffer) = depth_buffer {
            self.draw_triangles_multi_sampled(
                color_buffer,
                depth_buffer,
                uniform,
                chunk_size,
                buffer_height,
                &triangles,
            );
        } else {
            self.draw_triangles_no_depth_multi_sampled(
                color_buffer,
                uniform,
                chunk_size,
                buffer_height,
                &triangles,
            );
        }
    }
}
