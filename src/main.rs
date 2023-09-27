#![feature(new_uninit)]

mod texture;
use texture::*;

mod multi_sampled_texture;
use multi_sampled_texture::*;

mod sampler;
use sampler::*;

mod pipeline;
use pipeline::*;

mod font;
use font::*;

mod wgpu_state;
use wgpu_state::*;

use rayon::prelude::ParallelIterator;
use slender_math::*;
use std::path::Path;
use std::sync::Arc;

const RESOLUTION_DIVISOR: u32 = 1;
const DISPLAY_DEPTH: bool = false;
const DRAW_TEXT: bool = false;
const USE_MULTI_SAMPLING: bool = false;

#[derive(Clone, Copy)]
struct Vertex {
    position: v3f,
    normal: v3f,
    tex_coord: v2f,
    color: v3f,
}

#[derive(Clone, Copy)]
struct VertexOutput {
    normal: v3f,
    tex_coord: v2f,
    color: v4f,
}

impl pipeline::VertexOutput for VertexOutput {
    fn lerp(&self, rhs: &Self, t: f32) -> Self {
        Self {
            normal: self.normal.lerp(rhs.normal, t),
            tex_coord: self.tex_coord.lerp(rhs.tex_coord, t),
            color: self.color.lerp(rhs.color, t),
        }
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        Self {
            normal: self.normal + rhs.normal,
            tex_coord: self.tex_coord + rhs.tex_coord,
            color: self.color + rhs.color,
        }
    }

    #[inline]
    fn sub(&self, rhs: &Self) -> Self {
        Self {
            normal: self.normal - rhs.normal,
            tex_coord: self.tex_coord - rhs.tex_coord,
            color: self.color - rhs.color,
        }
    }

    #[inline]
    fn mul(&self, w: f32) -> Self {
        Self {
            normal: self.normal * w,
            tex_coord: self.tex_coord * w,
            color: self.color * w,
        }
    }
}

fn dummy_texture_srgb_4() -> ColorTexture<Srgb, 4> {
    let mut texture = ColorTexture::<Srgb, 4>::new(1, 1);
    texture.set_texel(0, 0, v4f::ONE);
    texture
}

fn dummy_texture_linear_4() -> ColorTexture<Linear, 4> {
    let mut texture = ColorTexture::<Linear, 4>::new(1, 1);
    texture.set_texel(0, 0, v4f::ONE);
    texture
}

fn dummy_texture_linear_1() -> ColorTexture<Linear, 1> {
    let mut texture = ColorTexture::<Linear, 1>::new(1, 1);
    texture.set_texel(0, 0, 1.0);
    texture
}

#[allow(dead_code)]
struct Material {
    ambient: v4f,
    diffuse: v4f,
    specular: v4f,
    shininess: f32,
    ambient_texture: ColorTexture<Srgb, 4>,
    diffuse_texture: ColorTexture<Srgb, 4>,
    specular_texture: ColorTexture<Linear, 1>,
    normal_texture: ColorTexture<Linear, 4>,
    shininess_texture: ColorTexture<Linear, 1>,
}

impl Default for Material {
    fn default() -> Self {
        let mut texture_argb_4 = ColorTexture::<Srgb, 4>::new(1, 1);
        texture_argb_4.set_texel(0, 0, v4f::ONE);

        Self {
            ambient: v4f::ONE,
            diffuse: v4f::new(1.0, 0.0, 1.0, 1.0),
            specular: v4f::ONE,
            shininess: 0.0,
            ambient_texture: dummy_texture_srgb_4(),
            diffuse_texture: dummy_texture_srgb_4(),
            specular_texture: dummy_texture_linear_1(),
            normal_texture: dummy_texture_linear_4(),
            shininess_texture: dummy_texture_linear_1(),
        }
    }
}

struct Uniform {
    view_proj: mat4,
    material: Arc<Material>,
    sampler: Sampler,
}

enum VS {}
impl VertexShader for VS {
    type Input = Vertex;
    type Output = VertexOutput;
    type Uniform = Uniform;

    fn run(input: Self::Input, uniform: &Self::Uniform) -> (v4f, Self::Output) {
        (
            uniform.view_proj * v4f::from_v3f(input.position, 1.0),
            VertexOutput {
                normal: input.normal,
                tex_coord: input.tex_coord,
                color: v4f::from_v3f(input.color, 1.0),
            },
        )
    }
}

enum PS {}
impl PixelShader for PS {
    type Input = VertexOutput;
    type Output = v4f;
    type Uniform = Uniform;

    fn run(
        input: Self::Input,
        dx: Self::Input,
        dy: Self::Input,
        uniform: &Self::Uniform,
    ) -> Self::Output {
        let diffuse = uniform.sampler.sample(
            &uniform.material.diffuse_texture,
            input.tex_coord,
            dx.tex_coord,
            dy.tex_coord,
        );
        diffuse * uniform.material.diffuse * input.color
    }
}

struct Mesh {
    vertices: Box<[Vertex]>,
    indices: Box<[u32]>,
    material_id: Option<usize>,
}

fn create_mesh(mesh: tobj::Mesh) -> Mesh {
    const DEFAULT_NORMAL: [f32; 3] = [0.0; 3];
    const DEFAULT_TEX_COORD: [f32; 2] = [0.0; 2];
    const DEFAULT_COLOR: [f32; 3] = [1.0; 3];

    let vertex_count = mesh.positions.len() / 3;
    let mut vertices = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let start3 = i * 3;
        let end3 = start3 + 3;
        let start2 = i * 2;
        let end2 = start2 + 2;

        let position = &mesh.positions[start3..end3];
        let normal = mesh.normals.get(start3..end3).unwrap_or(&DEFAULT_NORMAL);
        let tex_coord = mesh
            .texcoords
            .get(start2..end2)
            .unwrap_or(&DEFAULT_TEX_COORD);
        let color = mesh
            .vertex_color
            .get(start3..end3)
            .unwrap_or(&DEFAULT_COLOR);

        vertices.push(Vertex {
            position: v3f::new(position[0], position[1], position[2]) / 10.0,
            normal: v3f::new(normal[0], normal[1], normal[2]),
            tex_coord: v2f::new(tex_coord[0], 1.0 - tex_coord[1]),
            color: v3f::new(color[0], color[1], color[2]),
        })
    }

    Mesh {
        vertices: vertices.into_boxed_slice(),
        indices: mesh.indices.into_boxed_slice(),
        material_id: mesh.material_id,
    }
}

fn load_texture<CS: ColorSpace, const CHANNELS: usize>(
    path: &str,
    generate_mip_maps: bool,
) -> Option<ColorTexture<CS, CHANNELS>>
where
    Channels<CHANNELS>: ChannelDesc,
    ColorTexture<CS, CHANNELS>: LoadableTexture,
{
    let path = Path::join(
        "D:/Visual Studio Projekte/softpipe/".as_ref(),
        path.replace("\\\\", "/"),
    );

    match ColorTexture::<CS, CHANNELS>::load(path, generate_mip_maps) {
        Ok(texture) => Some(texture),
        Err(image::ImageError::IoError(_)) => None,
        Err(err) => panic!("{err}"),
    }
}

fn load_scene() -> (Vec<Mesh>, Vec<Arc<Material>>) {
    let (models, materials) = tobj::load_obj(
        "D:/Visual Studio Projekte/softpipe/models/sponza.obj",
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
    )
    .expect("Failed to load OBJ file");
    let materials = materials.expect("Failed to load MTL file");

    let meshes: Vec<_> = models
        .into_iter()
        .map(|model| create_mesh(model.mesh))
        .collect();

    let materials: Vec<_> = materials
        .into_iter()
        .map(|material| {
            Arc::new(Material {
                ambient: v4f::new(
                    material.ambient[0],
                    material.ambient[1],
                    material.ambient[2],
                    1.0,
                ),
                diffuse: v4f::new(
                    material.diffuse[0],
                    material.diffuse[1],
                    material.diffuse[2],
                    1.0,
                ),
                specular: v4f::new(
                    material.specular[0],
                    material.specular[1],
                    material.specular[2],
                    1.0,
                ),
                shininess: material.shininess,
                ambient_texture: load_texture::<Srgb, 4>(&material.ambient_texture, true)
                    .unwrap_or_else(dummy_texture_srgb_4),
                diffuse_texture: load_texture::<Srgb, 4>(&material.diffuse_texture, true)
                    .unwrap_or_else(dummy_texture_srgb_4),
                specular_texture: load_texture::<Linear, 1>(&material.specular_texture, true)
                    .unwrap_or_else(dummy_texture_linear_1),
                normal_texture: load_texture::<Linear, 4>(&material.normal_texture, true)
                    .unwrap_or_else(dummy_texture_linear_4),
                shininess_texture: load_texture::<Linear, 1>(&material.shininess_texture, true)
                    .unwrap_or_else(dummy_texture_linear_1),
            })
        })
        .collect();

    (meshes, materials)
}

fn main() {
    use winit::dpi::{PhysicalPosition, PhysicalSize};
    use winit::event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent};
    use winit::event_loop::EventLoop;
    use winit::window::WindowBuilder;

    let num_treads = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_treads)
        .build_global()
        .unwrap();

    const INITIAL_WIDTH: u32 = 1920;
    const INITIAL_HEIGHT: u32 = 1080;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("SoftPipe")
        .with_inner_size(PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
        .build(&event_loop)
        .unwrap();

    let mut wgpu_state = WgpuState::create(&window);

    let mut loop_helper = spin_sleep::LoopHelper::builder().build_with_target_rate(600.0);

    let mut pipeline = Pipeline::<VS, PS>::new(BlendMode::AlphaTest, FrontFace::Cw);
    let mut text_renderer = TextRenderer::new();

    let mut resolve_color_buffer = ColorTexture::<Linear, 4>::new(
        INITIAL_WIDTH / RESOLUTION_DIVISOR,
        INITIAL_HEIGHT / RESOLUTION_DIVISOR,
    );
    let mut resolve_depth_buffer = DepthTexture::new(
        INITIAL_WIDTH / RESOLUTION_DIVISOR,
        INITIAL_HEIGHT / RESOLUTION_DIVISOR,
    );

    let mut color_buffer = MultiSampledColorTexture::<Linear, 4>::new(
        INITIAL_WIDTH / RESOLUTION_DIVISOR,
        INITIAL_HEIGHT / RESOLUTION_DIVISOR,
    );
    let mut depth_buffer = MultiSampledDepthTexture::new(
        INITIAL_WIDTH / RESOLUTION_DIVISOR,
        INITIAL_HEIGHT / RESOLUTION_DIVISOR,
    );

    let default_material = Arc::new(Material::default());
    let mut uniform = Uniform {
        view_proj: mat4::IDENTITY,
        material: Arc::clone(&default_material),
        sampler: Sampler::new(TextureFilter::Linear, TextureWrap::Repeat),
    };

    let (meshes, materials) = load_scene();
    let msdf = Arc::new(load_texture::<Linear, 4>("fonts/Inter-Regular.png", false).unwrap());
    let font_file =
        std::fs::File::open("D:/Visual Studio Projekte/softpipe/fonts/Inter-Regular.json").unwrap();
    let font_file = std::io::BufReader::new(font_file);
    let font_atlas = Arc::new(FontAtlas::load(font_file, msdf.width(), msdf.height()).unwrap());
    let font = Font::new(font_atlas).with_weight(0.1);

    let mut window_focused = true;
    let mut forward = false;
    let mut back = false;
    let mut left = false;
    let mut right = false;
    let mut up = false;
    let mut down = false;
    let mut camera_pos = v3f::ZERO;
    let mut camera_angle_v = 0.0;
    let mut camera_angle_h = 0.0;
    let mut camera_dir = v3f::UNIT_Z;

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                }
                WindowEvent::Resized(size) => {
                    resolve_color_buffer.resize(
                        size.width / RESOLUTION_DIVISOR,
                        size.height / RESOLUTION_DIVISOR,
                    );
                    resolve_depth_buffer.resize(
                        size.width / RESOLUTION_DIVISOR,
                        size.height / RESOLUTION_DIVISOR,
                    );

                    if USE_MULTI_SAMPLING {
                        color_buffer.resize(
                            size.width / RESOLUTION_DIVISOR,
                            size.height / RESOLUTION_DIVISOR,
                        );
                        depth_buffer.resize(
                            size.width / RESOLUTION_DIVISOR,
                            size.height / RESOLUTION_DIVISOR,
                        );
                    }

                    wgpu_state.resize(size);
                }
                WindowEvent::KeyboardInput { input, .. } => match input.virtual_keycode {
                    Some(VirtualKeyCode::Escape) if input.state == ElementState::Pressed => {
                        control_flow.set_exit()
                    }
                    Some(VirtualKeyCode::W) => forward = input.state == ElementState::Pressed,
                    Some(VirtualKeyCode::S) => back = input.state == ElementState::Pressed,
                    Some(VirtualKeyCode::A) => left = input.state == ElementState::Pressed,
                    Some(VirtualKeyCode::D) => right = input.state == ElementState::Pressed,
                    Some(VirtualKeyCode::Space) => up = input.state == ElementState::Pressed,
                    _ => {}
                },
                WindowEvent::ModifiersChanged(state) => {
                    down = state.ctrl();
                }
                WindowEvent::Focused(focused) => {
                    window.set_cursor_visible(!focused);
                    window_focused = focused;
                }
                WindowEvent::CursorMoved { .. } => {
                    if window_focused {
                        window
                            .set_cursor_position(PhysicalPosition {
                                x: window.inner_size().width / 2,
                                y: window.inner_size().height / 2,
                            })
                            .unwrap();
                    }
                }
                _ => {}
            },
            Event::DeviceEvent {
                event:
                    DeviceEvent::MouseMotion {
                        delta: (delta_x, delta_y),
                    },
                ..
            } => {
                let delta_x = delta_x as f32;
                let delta_y = delta_y as f32;

                camera_angle_h += delta_x * 0.001;
                camera_angle_h %= std::f32::consts::TAU;

                camera_angle_v += delta_y * 0.001;
                camera_angle_v = camera_angle_v.clamp(
                    -std::f32::consts::FRAC_PI_2 * 0.9,
                    std::f32::consts::FRAC_PI_2 * 0.9,
                );

                camera_dir =
                    quat::from_yaw_pitch_roll(camera_angle_h, camera_angle_v, 0.0) * v3f::UNIT_Z;
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                loop_helper.loop_sleep();
                let elapsed = loop_helper.loop_start().as_secs_f32();

                const CAMERA_SPEED: f32 = 50.0;
                if forward {
                    camera_pos += camera_dir * (CAMERA_SPEED * elapsed);
                }
                if back {
                    camera_pos -= camera_dir * (CAMERA_SPEED * elapsed);
                }
                if left {
                    camera_pos +=
                        camera_dir.cross(v3f::UNIT_Y).normalized() * (CAMERA_SPEED * elapsed);
                }
                if right {
                    camera_pos -=
                        camera_dir.cross(v3f::UNIT_Y).normalized() * (CAMERA_SPEED * elapsed);
                }
                if up {
                    *camera_pos.y_mut() += CAMERA_SPEED * elapsed;
                }
                if down {
                    *camera_pos.y_mut() -= CAMERA_SPEED * elapsed;
                }

                if let Some(fps) = loop_helper.report_rate() {
                    window.set_title(&format!("SoftPipe - {fps:.1} fps"));
                }

                const CLEAR_COLOR: v4f = v4f::ZERO;
                const CLEAR_DEPTH: f32 = 1.0;
                if USE_MULTI_SAMPLING {
                    color_buffer
                    .par_rows_mut()
                    .for_each(|mut row| row.clear(CLEAR_COLOR));
                depth_buffer
                    .par_rows_mut()
                    .for_each(|mut row| row.clear(CLEAR_DEPTH));
                } else {
                    resolve_color_buffer
                    .par_rows_mut()
                    .for_each(|mut row| row.clear(CLEAR_COLOR));
                resolve_depth_buffer
                    .par_rows_mut()
                    .for_each(|mut row| row.clear(CLEAR_DEPTH));
                }

                let view = mat4::look_to(camera_pos, camera_dir, v3f::UNIT_Y);
                let proj = mat4::perspective(
                    std::f32::consts::FRAC_PI_2,
                    (color_buffer.width() as f32) / (color_buffer.height() as f32),
                    1.1,
                    500.0,
                );
                uniform.view_proj = proj * view;

                for mesh in meshes.iter() {
                    uniform.material = if let Some(material_id) = mesh.material_id {
                        Arc::clone(&materials[material_id])
                    } else {
                        Arc::clone(&default_material)
                    };

                    if USE_MULTI_SAMPLING {
                        pipeline.draw_indexed_multi_sampled(
                            &mut color_buffer,
                            Some(&mut depth_buffer),
                            &mesh.vertices,
                            &mesh.indices,
                            &uniform,
                        );
                    } else {
                        pipeline.draw_indexed(
                            &mut resolve_color_buffer,
                            Some(&mut resolve_depth_buffer),
                            &mesh.vertices,
                            &mesh.indices,
                            &uniform,
                        );
                    }
                }

                if USE_MULTI_SAMPLING {
                    color_buffer.resolve_into(&mut resolve_color_buffer);
                    depth_buffer.resolve_into(&mut resolve_depth_buffer);
                }

                if DRAW_TEXT {
                    let buffer_width = resolve_color_buffer.width();
                    text_renderer.draw_text(
                        &mut resolve_color_buffer,
                        "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",
                        Arc::clone(&msdf),
                        &font,
                        20.0,
                        v2f::ZERO,
                        v4f::ONE,
                        Some(buffer_width as f32),
                        Some(0.5),
                    );
                }

                if DISPLAY_DEPTH {
                    wgpu_state.render(&resolve_depth_buffer);
                } else {
                    wgpu_state.render(&resolve_color_buffer);
                }
            }
            Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
