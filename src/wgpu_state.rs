use crate::{DISPLAY_DEPTH, RESOLUTION_DIVISOR};
use bytemuck::{Pod, Zeroable};
use slender_math::*;
use wgpu::util::*;
use wgpu::*;

const TEXTURE_FORMAT: TextureFormat = if DISPLAY_DEPTH {
    TextureFormat::R32Float
} else {
    TextureFormat::Rgba8Unorm
};

const SHADER: &str = if DISPLAY_DEPTH {
    r#"
        struct VertexInput {
            @location(0) position: vec2<f32>,
            @location(1) tex_coord: vec2<f32>,
        };
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) tex_coord: vec2<f32>,
        };
        
        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4<f32>(input.position, 0.0, 1.0);
            output.tex_coord = input.tex_coord;
            return output;
        }

        @group(0)
        @binding(0)
        var tex: texture_2d<f32>;
        
        @group(0)
        @binding(1)
        var samp: sampler;
        
        @fragment
        fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
            var depth = textureSample(tex, samp, vertex.tex_coord).r;

            var A = 500.0 / (500.0 - 1.1);
            var B = -A * 1.1;
            var linear_depth = B / (depth - A) / 500.0;

            return vec4<f32>(linear_depth, linear_depth, linear_depth, 1.0);
        }
    "#
} else {
    r#"
        struct VertexInput {
            @location(0) position: vec2<f32>,
            @location(1) tex_coord: vec2<f32>,
        };

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) tex_coord: vec2<f32>,
        };

        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4<f32>(input.position, 0.0, 1.0);
            output.tex_coord = input.tex_coord;
            return output;
        }

        @group(0)
        @binding(0)
        var tex: texture_2d<f32>;

        @group(0)
        @binding(1)
        var samp: sampler;

        @fragment
        fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
            return textureSample(tex, samp, vertex.tex_coord);
        }
    "#
};

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct Vertex {
    position: v2f,
    tex_coord: v2f,
}

const fn vert(x: f32, y: f32, u: f32, v: f32) -> Vertex {
    Vertex {
        position: v2f::new(x, y),
        tex_coord: v2f::new(u, v),
    }
}

const TOP: f32 = 1.0;
const BOTTOM: f32 = -1.0;
const LEFT: f32 = -1.0;
const RIGHT: f32 = 1.0;
const VERTICES: [Vertex; 6] = [
    vert(LEFT, TOP, 0.0, 0.0),
    vert(LEFT, BOTTOM, 0.0, 1.0),
    vert(RIGHT, TOP, 1.0, 0.0),
    vert(RIGHT, TOP, 1.0, 0.0),
    vert(LEFT, BOTTOM, 0.0, 1.0),
    vert(RIGHT, BOTTOM, 1.0, 1.0),
];

#[allow(dead_code)]
pub struct WgpuState {
    instance: Instance,
    surface: Surface,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    shader: ShaderModule,
    swapchain_format: TextureFormat,
    bind_group_layout: BindGroupLayout,
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    texture: Texture,
    texture_view: TextureView,
    sampler: Sampler,
    bind_group: BindGroup,
}

impl WgpuState {
    pub fn create(window: &winit::window::Window) -> Self {
        let instance = Instance::default();

        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                label: None,
                features: Features::empty(),
                limits: Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits()),
            },
            None,
        ))
        .unwrap();

        let swapchain_caps = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_caps.formats[0];

        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: PresentMode::Mailbox,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_desc = BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        };
        let bind_group_layout = device.create_bind_group_layout(&bind_group_desc);

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &[
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        VertexAttribute {
                            format: VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&VERTICES),
            usage: BufferUsages::VERTEX,
        });

        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: window.inner_size().width / RESOLUTION_DIVISOR,
                height: window.inner_size().height / RESOLUTION_DIVISOR,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            instance,
            surface,
            adapter,
            device,
            queue,
            shader,
            swapchain_format,
            bind_group_layout,
            pipeline,
            vertex_buffer,
            texture,
            texture_view,
            sampler,
            bind_group,
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        let surface_config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: self.swapchain_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Mailbox,
            alpha_mode: CompositeAlphaMode::Opaque,
            view_formats: vec![],
        };
        self.surface.configure(&self.device, &surface_config);

        self.texture = self.device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: size.width / RESOLUTION_DIVISOR,
                height: size.height / RESOLUTION_DIVISOR,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.texture_view = self.texture.create_view(&TextureViewDescriptor::default());

        self.bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&self.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    pub fn render(&self, texture: &impl crate::Texture) {
        self.queue.write_texture(
            self.texture.as_image_copy(),
            texture.data(),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(
                    std::num::NonZeroU32::new(texture.width() * texture.bytes_per_texel()).unwrap(),
                ),
                rows_per_image: None,
            },
            Extent3d {
                width: texture.width(),
                height: texture.height(),
                depth_or_array_layers: 1,
            },
        );

        let frame = self.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}
