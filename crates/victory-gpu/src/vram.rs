//! A copy of title memory in video memory.
//!
//! [`crate::shadow`] keeps its copies in system memory ([`Host::shared_buffer`],
//! a `CUSTOM`/`L0` buffer): the title's memory is host memory, so the copy is
//! a duplicate of it that the CPU can refresh and compare, and every shader
//! read of it crosses PCIe. On the console the same read is GDDR6.
//!
//! This is the other arrangement, prepared and tested here and not wired in
//! yet: the copy shaders read is a buffer in video memory, brought up to date
//! by a copy per changed range (staged through upload memory) and read through
//! a readback buffer when the GPU wrote it.
//!
//! What the integration needs, in `shadow.rs` and `host.rs`:
//!
//! * `Shadow` gains `Option<VramCopy>`; `gpu()` returns its address where it
//!   has one. Video memory is worth it only for copies read far more than
//!   written and only within a budget - a copy of every title buffer would
//!   not fit beside the title's own resources;
//! * `Batch::flush` (which memcpys title memory into the mapped copy today)
//!   stages each range with [`VramCopy::upload`] instead, which records the
//!   copy on the list, so it runs before the list's shaders do;
//! * `Shadow::copy_back_page`, which compares the mapped copy, takes the page
//!   from [`VramCopy::read`] instead;
//! * the CPU copy stays: it is what the title's own threads read, what a
//!   refresh compares against, and what `crate::memory` counts.
//!
//! The tests are the semantics that has to hold for that to be safe: bytes
//! uploaded reach a shader that reads them by GPU address, what a shader
//! writes reads back, and two ranges of one list do not overwrite each other.

#![allow(dead_code)] // wired into `shadow` by the next change; the tests drive it.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::host::{Host, HostList, host, transition};

/// The state the copy is left in between its uses: common, which a shader's
/// root view and a copy both promote, and which every copy here enters and
/// leaves explicitly (a copy that starts and ends in it is what orders it
/// against the shader access on either side).
const COPY_BETWEEN: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_COMMON;

/// A copy of title memory in video memory.
pub(crate) struct VramCopy {
    buffer: ID3D12Resource,
    /// Where it is, in the title's address space.
    gpu: u64,
    len: u64,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for VramCopy {}
// SAFETY: as above.
unsafe impl Sync for VramCopy {}

impl VramCopy {
    /// `len` bytes of video memory (cut up to 64 KiB: copies and views start
    /// on whole elements), or `None` when the host will not give them.
    pub(crate) fn new(len: u64) -> Option<VramCopy> {
        let host = host()?;
        let len = len.max(1).next_multiple_of(1 << 16);
        let buffer = vram_buffer(host, len)?;
        crate::host::name(&buffer, || format!("title memory copy (video) {len:#x}"));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        (gpu != 0).then_some(VramCopy { buffer, gpu, len })
    }

    /// The address a shader reads this copy at.
    pub(crate) fn gpu(&self) -> u64 {
        self.gpu
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// `bytes` are title memory at `offset` in the copy: written to upload
    /// memory now, and copied into video memory when `list` runs (so the
    /// list's own shaders read them, and not before).
    ///
    /// `None` when the range does not fit or the host has no upload memory;
    /// nothing is recorded then.
    pub(crate) fn upload(&self, list: &HostList, offset: u64, bytes: &[u8]) -> Option<()> {
        let len = bytes.len() as u64;
        if len == 0 || offset.checked_add(len)? > self.len {
            return None;
        }
        let host = host()?;
        let (staging, mapped) = list.upload(host, len)?;
        // SAFETY: our mapped upload buffer of `len` bytes, kept by the list
        // until it is reset - after the copy has run.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
        // SAFETY: an open list and live buffers. The copy destination is
        // named and given back, as a copy of an indirect argument buffer is:
        // a copy between two states is what orders it against the shader
        // access before and after (a bare copy from common is not).
        unsafe {
            let l = &list.list;
            l.ResourceBarrier(&[transition(&self.buffer, COPY_BETWEEN, D3D12_RESOURCE_STATE_COPY_DEST)]);
            l.CopyBufferRegion(&self.buffer, offset, &staging, 0, len);
            l.ResourceBarrier(&[transition(&self.buffer, D3D12_RESOURCE_STATE_COPY_DEST, COPY_BETWEEN)]);
        }
        Some(())
    }

    /// A readback of `[offset, offset + len)` as `list` leaves it, read after
    /// the list has run ([`Read::bytes`]).
    pub(crate) fn read(&self, list: &HostList, offset: u64, len: u64) -> Option<Read> {
        if len == 0 || offset.checked_add(len)? > self.len {
            return None;
        }
        let host = host()?;
        let (readback, mapped) = list.readback(host, len)?;
        // SAFETY: an open list and live buffers, as above: the copy source is
        // named and given back, so the read sees what ran before it.
        unsafe {
            let l = &list.list;
            l.ResourceBarrier(&[transition(&self.buffer, COPY_BETWEEN, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyBufferRegion(&readback, 0, &self.buffer, offset, len);
            l.ResourceBarrier(&[transition(&self.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, COPY_BETWEEN)]);
        }
        Some(Read { buffer: readback, data: mapped.cast(), len })
    }
}

/// A recorded readback: the bytes once the list that recorded it has run.
pub(crate) struct Read {
    /// Held so the address names nothing else before the read.
    buffer: ID3D12Resource,
    data: *mut u8,
    len: u64,
}

// SAFETY: D3D12 resources are free-threaded; `data` is read after the list
// that wrote it has run.
unsafe impl Send for Read {}

impl Read {
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// The bytes the GPU left in the range.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        // SAFETY: a mapped readback buffer of `len` bytes, written by the
        // copy the list recorded.
        unsafe { std::slice::from_raw_parts(self.data, self.len as usize) }.to_vec()
    }
}

/// A buffer in video memory that shaders read and write: unordered access
/// allowed, common, so a copy and a root view promote it as they need.
fn vram_buffer(host: &Host, size: u64) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a committed buffer.
    let made = unsafe {
        host.device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
    };
    made.ok().and(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dst = src * 2` and `dst + 4 = src + 4 + 1`, so a read says both which
    /// bytes arrived and that the shader, not the upload, wrote them.
    const SHADER: &str = r#"
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), SRV(t0), UAV(u0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    dst.Store(0, src.Load(0) * 2);
    dst.Store(4, src.Load(4) + 1);
}
"#;

    /// `bytes` as little-endian words: what the tests compare.
    fn words(bytes: &[u8]) -> Vec<u32> {
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks(4) {
            out.push(u32::from_le_bytes(chunk.try_into().expect("4")));
        }
        out
    }

    struct Copy {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
    }

    /// The test's compute pipeline, or `None` where the host has no GPU or
    /// no shader compiler.
    fn pipeline() -> Option<Copy> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        let host = host()?;
        let (mut code, mut errors) = (None::<ID3DBlob>, None::<ID3DBlob>);
        // SAFETY: the source lives through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                SHADER.as_ptr().cast(),
                SHADER.len(),
                s!("copy"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        compiled.ok()?;
        let code = code?;
        // SAFETY: the blob's own bytes; the root signature is in the shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).ok()?;
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            Some(Copy { root, pso })
        }
    }

    /// Records the test's dispatch over `src` and `dst`.
    fn dispatch(list: &HostList, copy: &Copy, src: u64, dst: u64) {
        let l = &list.list;
        // SAFETY: an open list; our root signature, pipeline and buffers.
        unsafe {
            l.SetComputeRootSignature(&copy.root);
            l.SetPipelineState(&copy.pso);
            l.SetComputeRootShaderResourceView(0, src);
            l.SetComputeRootUnorderedAccessView(1, dst);
            l.Dispatch(1, 1, 1);
        }
    }

    /// Runs `record` on one list and returns what `read` gives after it.
    fn run<T>(record: impl FnOnce(&HostList, &Copy) -> T) -> Option<T> {
        let host = host()?;
        let copy = pipeline()?;
        let list = host.new_list().ok()?;
        let out = record(&list, &copy);
        list.close();
        host.execute(&[&list]);
        Some(out)
    }

    /// What the host uploads reaches a shader reading the copy at its GPU
    /// address, and the shader's result reads back: the two ends of the
    /// arrangement, in one run.
    #[test]
    fn uploaded_bytes_reach_a_shader_and_its_result_reads_back() {
        let Some((src, dst)) = host().and_then(|_| Some((VramCopy::new(64 << 10)?, VramCopy::new(64 << 10)?)))
        else {
            return; // no host GPU here
        };
        let pattern: [u32; 2] = [7, 41];
        let bytes: Vec<u8> = pattern.iter().flat_map(|v| v.to_le_bytes()).collect();
        let read = run(|list, copy| {
            src.upload(list, 128, &bytes).expect("an upload");
            dispatch(list, copy, src.gpu() + 128, dst.gpu());
            // Recorded after the dispatch: this is what the shader left.
            dst.read(list, 0, 8).expect("a readback")
        });
        let Some(read) = read else {
            return; // no shader compiler here either
        };
        let out = words(&read.bytes());
        assert_eq!(out, [14, 42], "the shader read what the host uploaded");
    }

    /// What a shader writes into the copy is what a later read gives - the
    /// write-back path's view of the page.
    #[test]
    fn what_a_shader_writes_reads_back() {
        let Some((src, dst)) = host().and_then(|_| Some((VramCopy::new(64 << 10)?, VramCopy::new(64 << 10)?)))
        else {
            return;
        };
        let pattern: [u32; 2] = [100, 9];
        let bytes: Vec<u8> = pattern.iter().flat_map(|v| v.to_le_bytes()).collect();
        let read = run(|list, copy| {
            src.upload(list, 0, &bytes).expect("an upload");
            dispatch(list, copy, src.gpu(), dst.gpu() + 512);
            dst.read(list, 512, 8).expect("a readback")
        });
        let Some(read) = read else {
            return;
        };
        let out = words(&read.bytes());
        assert_eq!(out, [200, 10]);
    }

    /// The upload and the readback alone, with no shader between them: the
    /// bytes go into video memory and come back.
    #[test]
    fn an_upload_reads_back_without_a_shader() {
        let Some(copy) = host().and_then(|_| VramCopy::new(64 << 10)) else {
            return;
        };
        let host = super::host().expect("a host GPU");
        let list = host.new_list().expect("a list");
        let bytes: Vec<u8> = (0..64u32).flat_map(|v| v.wrapping_mul(7).to_le_bytes()).collect();
        copy.upload(&list, 256, &bytes).expect("an upload");
        let read = copy.read(&list, 256, bytes.len() as u64).expect("a readback");
        list.close();
        host.execute(&[&list]);
        assert_eq!(read.bytes(), bytes, "the bytes of the copy");
    }

    /// Two ranges of one list land where they were asked for: a staged upload
    /// is not shared behind the caller's back.
    #[test]
    fn two_ranges_in_one_list_do_not_overwrite_each_other() {
        let Some(copy) = host().and_then(|_| VramCopy::new(64 << 10)) else {
            return;
        };
        let (first, second) = (0xdead_beef_u32.to_le_bytes(), 0x1234_5678_u32.to_le_bytes());
        let host = super::host().expect("a host GPU");
        let list = host.new_list().expect("a list");
        copy.upload(&list, 0, &first).expect("the first range");
        copy.upload(&list, 4096, &second).expect("the second range");
        let reads = (copy.read(&list, 0, 4).expect("read one"), copy.read(&list, 4096, 4).expect("read two"));
        list.close();
        host.execute(&[&list]);
        assert_eq!(reads.0.bytes(), first);
        assert_eq!(reads.1.bytes(), second);
    }

    /// Read kernels for the measurement below: `seq` walks its own 64 bytes
    /// of the copy, `rnd` takes eight pseudo-random 64-byte lines of it.
    const BENCH: &str = r#"
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer Size : register(b0) { uint lines; };
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void seq(uint3 id : SV_DispatchThreadID) {
    uint4 v = src.Load4(id.x * 16);
    if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9e3779b9u) {
        dst.Store(0, id.x);
    }
}
[RootSignature(RS)]
[numthreads(256, 1, 1)]
void rnd(uint3 id : SV_DispatchThreadID) {
    uint h = id.x * 747796405u + 2891336453u;
    h = h * 1664525u + 1013904223u;
    uint4 v = src.Load4(((h >> 8) % (lines * 4)) * 16);
    if ((v.x ^ v.y ^ v.z ^ v.w) == 0x9e3779b9u) {
        dst.Store(0, id.x);
    }
}
"#;

    /// The test's two kernels, compiled once.
    fn bench_pipelines() -> Option<(ID3D12RootSignature, [ID3D12PipelineState; 2])> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        let host = host()?;
        let mut blobs: [Option<ID3DBlob>; 2] = [None, None];
        for (i, entry) in ["seq", "rnd"].into_iter().enumerate() {
            let name = std::ffi::CString::new(entry).ok()?;
            let mut code = None;
            // SAFETY: the source lives through the call; out pointers ours.
            let made = unsafe {
                D3DCompile(
                    BENCH.as_ptr().cast(),
                    BENCH.len(),
                    s!("bench"),
                    None,
                    None,
                    windows::core::PCSTR(name.as_ptr().cast()),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )
            };
            made.ok()?;
            blobs[i] = code;
        }
        // SAFETY: the blobs' own bytes; the root signature is in them.
        unsafe {
            let bytes = |b: &ID3DBlob| {
                std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
            };
            let first = blobs[0].as_ref()?;
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes(first)).ok()?;
            let pipe = |b: &ID3DBlob| {
                host.device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes(b).as_ptr().cast(),
                            BytecodeLength: bytes(b).len(),
                        },
                        ..Default::default()
                    })
                    .ok()
            };
            let first = pipe(blobs[0].as_ref()?)?;
            let second = pipe(blobs[1].as_ref()?)?;
            Some((root, [first, second]))
        }
    }

    /// What reading a copy costs where it lives: a stride-1 scan and a
    /// random line a thread, over the same 128 MiB, once in video
    /// memory and once in the system memory `shadow`'s copies are in today.
    ///
    /// Not an assertion: the ratio is the machine's (and a card with resize
    /// BAR flatters the system-memory side). It is here because it answers, in
    /// seconds and without the title, whether putting these copies in video
    /// memory is worth the integration - the one experiment so far
    /// (`VICTORY_COPIES_IN_VIDEO_MEMORY`) used a BAR window and measured
    /// worse, which says nothing either way.
    ///
    ///     cargo test -p victory-gpu --release the_read_cost -- --nocapture
    #[test]
    fn the_read_cost_of_a_copy_where_it_lives() {
        const SIZE: u64 = 128 << 20;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some((root, psos)) = bench_pipelines() else {
            return; // no shader compiler here
        };
        let lines = (SIZE / 64) as u32;
        // One 16-byte read a thread: 256 threads a group.
        let groups = (SIZE / 16 / 256) as u32;
        let Some(video) = VramCopy::new(SIZE) else { return };
        let Some((system, system_data)) = host.shared_buffer(SIZE) else { return };
        // SAFETY: a live buffer.
        let system_gpu = unsafe { system.GetGPUVirtualAddress() };
        // What a kernel writes if it reads a sentinel: nothing, in practice,
        // so the timing is of the reads alone.
        let Some((out, out_data)) = host.shared_buffer(4096) else { return };
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        let pattern: Vec<u8> = (0..SIZE as usize).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        // SAFETY: our mapped copy of `SIZE` bytes, written before it is read.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), system_data, SIZE as usize) };

        // The upload is not part of a read's cost, so it runs once, before
        // the timed lists (the integration's own uploads are incremental).
        {
            let Ok(list) = host.new_list() else { return };
            if video.upload(&list, 0, &pattern).is_none() {
                return;
            }
            list.close();
            host.execute(&[&list]);
        }

        // Ten dispatches a list: one is 0.3 ms of reads and as much again of
        // submission, and the ratio wanted is the reads'.
        const REPS: usize = 10;
        let time = |src: u64, kernel: usize| -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                let Ok(list) = host.new_list() else { return 0.0 };
                // SAFETY: an open list; our pipeline and buffers.
                unsafe {
                    let l = &list.list;
                    l.SetComputeRootSignature(&root);
                    l.SetPipelineState(&psos[kernel]);
                    l.SetComputeRoot32BitConstant(0, lines, 0);
                    l.SetComputeRootShaderResourceView(1, src);
                    l.SetComputeRootUnorderedAccessView(2, out_gpu);
                    for _ in 0..REPS {
                        l.Dispatch(groups, 1, 1);
                    }
                }
                list.close();
                let started = std::time::Instant::now();
                host.execute(&[&list]);
                best = best.min(started.elapsed().as_secs_f64() * 1000.0);
            }
            best / REPS as f64
        };
        let (seq_video, seq_system) = (time(video.gpu(), 0), time(system_gpu, 0));
        let (rnd_video, rnd_system) = (time(video.gpu(), 1), time(system_gpu, 1));
        // SAFETY: our mapped output, read after the lists ran.
        let sink = unsafe { std::slice::from_raw_parts(out_data.cast::<u32>(), 4) };
        println!("\n  {:>14} {:>10} {:>10} {:>8}", "128 MiB read", "video", "system", "ratio");
        println!(
            "  {:>14} {seq_video:>8.3}ms {seq_system:>8.3}ms {:>7.2}x",
            "stride-1 scan",
            seq_system / seq_video.max(1e-9)
        );
        println!(
            "  {:>14} {rnd_video:>8.3}ms {rnd_system:>8.3}ms {:>7.2}x",
            "random lines",
            rnd_system / rnd_video.max(1e-9)
        );
        println!("  (sentinel hits: {sink:?})\n");
    }

    /// A range that does not fit the copy is refused, and nothing is
    /// recorded for it.
    #[test]
    fn a_range_past_the_end_is_refused() {
        let Some(copy) = host().and_then(|_| VramCopy::new(1 << 16)) else {
            return;
        };
        let host = super::host().expect("a host GPU");
        let list = host.new_list().expect("a list");
        let len = copy.len();
        let byte = [0u8; 4];
        assert!(copy.upload(&list, len - 4, &byte).is_some(), "the last element fits");
        assert!(copy.upload(&list, len - 2, &byte).is_none(), "one past the end");
        assert!(copy.upload(&list, 0, &[]).is_none(), "nothing to copy");
        assert!(copy.read(&list, len - 4, 8).is_none(), "past the end");
        assert!(copy.read(&list, 0, 0).is_none(), "nothing to read");
        list.close();
    }

    /// A copy's length is a whole number of 64 KiB pages, which is what
    /// views and copies inside it start on.
    #[test]
    fn a_copy_is_whole_pages() {
        let Some(copy) = host().and_then(|_| VramCopy::new(1)) else {
            return;
        };
        assert_eq!(copy.len(), 1 << 16);
        assert_eq!(copy.gpu() & 0xff, 0, "a buffer's address is 256-byte aligned");
    }
}
