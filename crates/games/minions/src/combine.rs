//! Cross-worker batching of ORT calls ([`Combiner`]), and the device-resident
//! trunk-feature store ([`HStore`]) the store-indexed head graphs read from.
//!
//! Both exist because self-play's head calls were bound by per-call overhead
//! and by bytes over PCIe, not by arithmetic: see `ort_net`'s `Pinned` for the
//! measurements that led here.

use std::sync::{Condvar, Mutex};

use half::f16;
use ort::session::Session;

use crate::map::{N, NUM_CELLS};

fn e2a<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

/// A port's element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elem {
    F16,
    F32,
    I64,
}

impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F16 => 2,
            Elem::F32 => 4,
            Elem::I64 => 8,
        }
    }

    fn onnx(self) -> ort::sys::ONNXTensorElementDataType {
        use ort::sys::ONNXTensorElementDataType as T;
        match self {
            Elem::F16 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            Elem::F32 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            Elem::I64 => T::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        }
    }

    fn of(t: ort::value::TensorElementType) -> anyhow::Result<Self> {
        use ort::value::TensorElementType as T;
        Ok(match t {
            T::Float16 => Elem::F16,
            T::Float32 => Elem::F32,
            T::Int64 => Elem::I64,
            other => anyhow::bail!("unsupported port type {other:?}"),
        })
    }
}

/// Which leading dimension a port's rows count: the call's positions, or the
/// gathered logit rows of a store-indexed diffusion call (`cell_flat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    Rows,
    Cells,
}

/// One graph input or output the combiner stages.
#[derive(Clone, Debug)]
pub(crate) struct Port {
    pub elem: Elem,
    pub axis: Axis,
    /// Dims after the leading one, all static.
    pub tail: Vec<i64>,
}

impl Port {
    /// Elements per leading-dimension row.
    pub fn stride(&self) -> usize {
        self.tail.iter().product::<i64>().max(1) as usize
    }

    fn row_bytes(&self) -> usize {
        self.stride() * self.elem.size()
    }
}

/// A port's slice of one job's window, typed by the caller.
pub(crate) struct Slot<'a> {
    pub bytes: &'a mut [u8],
}

impl Slot<'_> {
    pub fn f16(&mut self) -> &mut [f16] {
        // SAFETY: arena regions are allocated by ORT's pinned allocator (256-byte
        // aligned) and windows start at whole rows of this element type.
        unsafe {
            std::slice::from_raw_parts_mut(self.bytes.as_mut_ptr().cast(), self.bytes.len() / 2)
        }
    }

    pub fn i64(&mut self) -> &mut [i64] {
        // SAFETY: as `f16`.
        unsafe {
            std::slice::from_raw_parts_mut(self.bytes.as_mut_ptr().cast(), self.bytes.len() / 8)
        }
    }
}

/// A finished job's view of its output rows.
pub(crate) struct Out<'a> {
    pub bytes: &'a [u8],
}

impl Out<'_> {
    pub fn f16(&self) -> &[f16] {
        // SAFETY: as `Slot::f16`.
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.bytes.len() / 2) }
    }

    pub fn f32(&self) -> &[f32] {
        // SAFETY: as `Slot::f16`.
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().cast(), self.bytes.len() / 4) }
    }
}

/// An ORT tensor over memory the caller owns: the value is released on drop,
/// the data never. Unlike `TensorRefMut::from_raw`, the result can go to
/// `OutputSelector::preallocate`, which wants an owned value.
///
/// # Safety
/// `data` must be valid for `shape` and outlive the tensor, and `info` must
/// describe where it lives.
pub(crate) unsafe fn tensor_over(
    info: &ort::memory::MemoryInfo<'static>,
    data: *mut u8,
    elem: Elem,
    shape: &[i64],
) -> anyhow::Result<ort::value::DynValue> {
    use ort::AsPointer;
    let n: i64 = shape.iter().product();
    let mut out: *mut ort::sys::OrtValue = std::ptr::null_mut();
    let st = unsafe {
        (ort::api().CreateTensorWithDataAsOrtValue)(
            info.ptr(),
            data.cast(),
            n.max(0) as usize * elem.size(),
            shape.as_ptr(),
            shape.len(),
            elem.onnx(),
            &mut out,
        )
    };
    if !st.0.is_null() {
        unsafe { (ort::api().ReleaseStatus)(st.0) };
        anyhow::bail!("CreateTensorWithDataAsOrtValue failed for {elem:?} {shape:?}");
    }
    let ptr = std::ptr::NonNull::new(out).ok_or_else(|| anyhow::anyhow!("null OrtValue"))?;
    Ok(unsafe { ort::value::Value::from_ptr(ptr, None) })
}

/// Runs one combined batch: the input and output tensors, in port order, and
/// the batch's row and cell counts.
/// Runs one batch: inputs, preallocated outputs, rows, cells, and the runner
/// slot it holds -- `0..runners`, one session each, so concurrent batches
/// never share a session.
pub(crate) type RunFn<'a> = dyn Fn(
        Vec<ort::value::DynValue>,
        Vec<ort::value::DynValue>,
        usize,
        usize,
        usize,
    ) -> anyhow::Result<()>
    + 'a;

/// One of a [`Combiner`]'s pinned staging areas: every port's region.
struct Arena {
    /// Inputs, then outputs, each `cap` rows (of its axis) long.
    bufs: Vec<*mut u8>,
    rows: usize,
    cells: usize,
    jobs: usize,
    filled: usize,
    /// Jobs of the finished batch still to read their output.
    readers: usize,
    /// Batches this arena has run; a job reserved at `gen` is done after it.
    gen: u64,
    err: Option<String>,
    /// Closed: on the GPU, or with readers left.
    busy: bool,
}

impl Arena {
    fn reset(&mut self) {
        (self.rows, self.cells, self.jobs, self.filled) = (0, 0, 0, 0);
    }
}

struct State {
    arenas: Vec<Arena>,
    /// The arena taking reservations; `None` while every one is busy.
    open: Option<usize>,
    /// Runner slots in use, one bit each.
    slots: u32,
}

impl State {
    /// Open an idle arena for reservations, if there is one.
    fn open_idle(&mut self) {
        if let Some(i) = self.arenas.iter().position(|a| !a.busy) {
            self.arenas[i].reset();
            self.open = Some(i);
        }
    }
}

/// Cross-worker batching for one graph -- the replacement for a mutex.
///
/// With a mutex, six workers' calls ran one after another: one worker's rows
/// on the GPU, five waiting, then the next. A call costs ~150 us before its
/// first row, and TensorRT's throughput per row keeps rising well past the
/// ~160 rows one worker brings, so the queue itself was the waste. Here a
/// worker *reserves* rows in the open arena and fills them in parallel with
/// everyone else; whoever finds no batch running closes that arena, opens the
/// other, and runs everything reserved as one call. Each worker then reads its
/// own slice of the output. The only serial work left is the GPU call.
///
/// Up to `runners` batches run at once, one session each, so the host work
/// of one call (binding, enqueue, the wait) overlaps another's kernels. There
/// is one arena more than runners, so reservation never waits on a running
/// batch: while the others are on the GPU (or being read), one fills. An arena
/// reopens only once all its readers are done.
pub(crate) struct Combiner {
    st: Mutex<State>,
    cv: Condvar,
    runners: usize,
    cap_rows: usize,
    cap_cells: usize,
    ins: Vec<Port>,
    outs: Vec<Port>,
    input_info: ort::memory::MemoryInfo<'static>,
    output_info: ort::memory::MemoryInfo<'static>,
    /// The pinned allocations the arenas point into, kept alive.
    _backing: Vec<ort::value::Tensor<u8>>,
}

// SAFETY: the raw pointers are into `_backing`, which lives as long as the
// combiner; every region is written by one thread at a time under the protocol
// in `submit` (reserved rows by their job, outputs by the runner, before the
// readers are released).
unsafe impl Send for Combiner {}
unsafe impl Sync for Combiner {}

/// A port named `name` of `sess`, as `(elem, tail)`; every dim after the first
/// must be static.
fn port_of(sess: &Session, name: &str, axis: Axis, input: bool) -> anyhow::Result<Port> {
    let list = if input { sess.inputs() } else { sess.outputs() };
    let o = list.iter().find(|o| o.name() == name).ok_or_else(|| {
        anyhow::anyhow!(
            "graph has no {} {name}",
            if input { "input" } else { "output" }
        )
    })?;
    let ort::value::ValueType::Tensor { ty, shape, .. } = o.dtype() else {
        anyhow::bail!("{name} is not a tensor");
    };
    let tail: Vec<i64> = shape.iter().skip(1).copied().collect();
    anyhow::ensure!(
        tail.iter().all(|&d| d > 0),
        "{name}: only the leading dim may be dynamic ({shape:?})"
    );
    Ok(Port {
        elem: Elem::of(*ty)?,
        axis,
        tail,
    })
}

impl Combiner {
    /// A combiner for `sess`'s ports `ins` and `outs`, `cap_rows` rows (and
    /// `cap_cells` gathered cells) per batch.
    pub fn new(
        sess: &Session,
        ins: &[(&str, Axis)],
        outs: &[(&str, Axis)],
        cap_rows: usize,
        cap_cells: usize,
        runners: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=32).contains(&runners), "{runners} runners");
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        let ins: Vec<Port> = ins
            .iter()
            .map(|&(n, a)| port_of(sess, n, a, true))
            .collect::<anyhow::Result<_>>()?;
        let outs: Vec<Port> = outs
            .iter()
            .map(|&(n, a)| port_of(sess, n, a, false))
            .collect::<anyhow::Result<_>>()?;
        let mi = |t| MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, t);
        let (input_info, output_info) = (
            mi(MemoryType::CPUInput).map_err(e2a)?,
            mi(MemoryType::CPUOutput).map_err(e2a)?,
        );
        let ain = Allocator::new(sess, input_info.clone()).map_err(e2a)?;
        let aout = Allocator::new(sess, output_info.clone()).map_err(e2a)?;
        let cap = |a: Axis| if a == Axis::Rows { cap_rows } else { cap_cells };
        let mut backing = Vec::new();
        let mut arena = || -> anyhow::Result<Arena> {
            let mut bufs = Vec::new();
            for (p, alloc) in ins
                .iter()
                .map(|p| (p, &ain))
                .chain(outs.iter().map(|p| (p, &aout)))
            {
                let n = (cap(p.axis) * p.row_bytes()).max(8);
                let mut t = ort::value::Tensor::<u8>::new(alloc, [n]).map_err(e2a)?;
                bufs.push(
                    t.try_extract_tensor_mut::<u8>()
                        .map_err(e2a)?
                        .1
                        .as_mut_ptr(),
                );
                backing.push(t);
            }
            Ok(Arena {
                bufs,
                rows: 0,
                cells: 0,
                jobs: 0,
                filled: 0,
                readers: 0,
                gen: 0,
                err: None,
                busy: false,
            })
        };
        let arenas = (0..=runners)
            .map(|_| arena())
            .collect::<anyhow::Result<_>>()?;
        // The allocators must outlive the buffers; like the sessions they came
        // from, they are leaked.
        std::mem::forget(ain);
        std::mem::forget(aout);
        Ok(Self {
            st: Mutex::new(State {
                arenas,
                open: Some(0),
                slots: 0,
            }),
            cv: Condvar::new(),
            runners,
            cap_rows,
            cap_cells,
            ins,
            outs,
            input_info,
            output_info,
            _backing: backing,
        })
    }

    /// Tensors over arena `bufs`' first `rows`/`cells` rows, inputs then
    /// outputs.
    ///
    /// # Safety
    /// The batch must own those rows until the tensors are dropped.
    unsafe fn tensors(
        &self,
        bufs: &[*mut u8],
        rows: usize,
        cells: usize,
    ) -> anyhow::Result<(Vec<ort::value::DynValue>, Vec<ort::value::DynValue>)> {
        let lead = |a: Axis| (if a == Axis::Rows { rows } else { cells }) as i64;
        let mk = |p: &Port, buf: *mut u8, info: &ort::memory::MemoryInfo<'static>| {
            let shape: Vec<i64> = std::iter::once(lead(p.axis))
                .chain(p.tail.iter().copied())
                .collect();
            unsafe { tensor_over(info, buf, p.elem, &shape) }
        };
        let ins = self
            .ins
            .iter()
            .zip(bufs)
            .map(|(p, &b)| mk(p, b, &self.input_info))
            .collect::<anyhow::Result<_>>()?;
        let outs = self
            .outs
            .iter()
            .zip(&bufs[self.ins.len()..])
            .map(|(p, &b)| mk(p, b, &self.output_info))
            .collect::<anyhow::Result<_>>()?;
        Ok((ins, outs))
    }

    /// Run one batch at the caps with zeroed inputs, before any real job.
    ///
    /// TensorRT's EP widens an engine's shape range whenever a call falls
    /// outside it, and every widening is a full rebuild. Combined batches grow
    /// in small steps as load varies, so without this each new largest batch
    /// would stall the actor for a rebuild -- the failure that once took
    /// self-play from 1605 to 52 expansions/s. One call at the caps widens the
    /// range once, up front. Zero is a valid index for every index port.
    ///
    /// Run at both ends of the range, one row (and cell) and the caps: an
    /// engine first built at one shape takes the other as a widening too. And
    /// on every runner slot, since each has its own session and engine.
    pub fn warm(&self, run: &RunFn<'_>) -> anyhow::Result<()> {
        let st = self.st.lock().unwrap();
        let ar = &st.arenas[0];
        let cap = |a: Axis| {
            if a == Axis::Rows {
                self.cap_rows
            } else {
                self.cap_cells
            }
        };
        for (p, &b) in self.ins.iter().zip(&ar.bufs) {
            // SAFETY: nothing has been reserved yet (the lock is held), and the
            // region is `cap` rows long.
            unsafe { std::slice::from_raw_parts_mut(b, cap(p.axis) * p.row_bytes()).fill(0) };
        }
        for slot in 0..self.runners {
            for (rows, cells) in [(1, self.cap_cells.min(1)), (self.cap_rows, self.cap_cells)] {
                // SAFETY: as above.
                let (ins, outs) = unsafe { self.tensors(&ar.bufs, rows, cells)? };
                run(ins, outs, rows, cells, slot)?;
            }
        }
        Ok(())
    }

    /// One job through the next combined call: `b` rows and `p` gathered
    /// cells. `fill` writes the job's inputs, given its row and cell offsets
    /// in the batch (a cell index names a batch row, so it must add the
    /// former) and one slot per input port; `read` then receives one view per
    /// output port.
    pub fn submit(
        &self,
        b: usize,
        p: usize,
        fill: &mut dyn FnMut(usize, usize, &mut [Slot<'_>]) -> anyhow::Result<()>,
        read: &mut dyn FnMut(&[Out<'_>]) -> anyhow::Result<()>,
        run: &RunFn<'_>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            b <= self.cap_rows && p <= self.cap_cells,
            "job of {b} rows / {p} cells over the caps"
        );
        let mut st = self.st.lock().unwrap();
        // Reserve.
        let (a, roff, coff, gen) = loop {
            if let Some(a) = st.open {
                let ar = &mut st.arenas[a];
                if ar.rows + b <= self.cap_rows && ar.cells + p <= self.cap_cells {
                    let (r, c) = (ar.rows, ar.cells);
                    ar.rows += b;
                    ar.cells += p;
                    ar.jobs += 1;
                    break (a, r, c, ar.gen);
                }
            }
            st = self.cv.wait(st).unwrap();
        };
        let bufs = st.arenas[a].bufs.clone();
        drop(st);
        // Fill, in parallel with every other job.
        let filled = {
            // SAFETY: this job's rows/cells of arena `a` are its alone until it
            // reports them filled.
            let mut slots: Vec<Slot<'_>> = self
                .ins
                .iter()
                .zip(&bufs)
                .map(|(port, &buf)| {
                    let (off, n) = if port.axis == Axis::Rows {
                        (roff, b)
                    } else {
                        (coff, p)
                    };
                    Slot {
                        bytes: unsafe {
                            std::slice::from_raw_parts_mut(
                                buf.add(off * port.row_bytes()),
                                n * port.row_bytes(),
                            )
                        },
                    }
                })
                .collect();
            fill(roff, coff, &mut slots)
        };
        let mut st = self.st.lock().unwrap();
        st.arenas[a].filled += 1;
        if let Err(e) = &filled {
            // The batch still runs, over this job's garbage; only this job
            // reports the error.
            eprintln!("combiner fill failed: {e:#}");
        }
        self.cv.notify_all();
        // Wait for a batch to carry this job, running it if a runner slot is
        // free. With the pool off (`pool_ab`) only one batch runs at a time.
        let max = if crate::ort_net::POOL_USE.load(std::sync::atomic::Ordering::Relaxed) {
            self.runners
        } else {
            1
        };
        while st.arenas[a].gen == gen {
            let slot = (!st.slots).trailing_zeros() as usize;
            if slot < max && st.open == Some(a) {
                st.slots |= 1 << slot;
                st.open = None;
                st.arenas[a].busy = true;
                st.open_idle();
                self.cv.notify_all();
                while st.arenas[a].filled < st.arenas[a].jobs {
                    st = self.cv.wait(st).unwrap();
                }
                let (rows, cells) = (st.arenas[a].rows, st.arenas[a].cells);
                drop(st);
                // SAFETY: the batch owns rows `..rows`/`..cells` of arena `a`
                // until its readers are released, after the tensors are gone.
                let res = unsafe { self.tensors(&bufs, rows, cells) }
                    .and_then(|(ins, outs)| run(ins, outs, rows, cells, slot));
                st = self.st.lock().unwrap();
                let ar = &mut st.arenas[a];
                ar.err = res.err().map(|e| format!("{e:#}"));
                ar.readers = ar.jobs;
                ar.gen += 1;
                st.slots &= !(1 << slot);
                self.cv.notify_all();
                break;
            }
            st = self.cv.wait(st).unwrap();
        }
        let err = st.arenas[a].err.clone();
        drop(st);
        let res = match (err, filled) {
            (Some(e), _) => Err(anyhow::anyhow!("combined call failed: {e}")),
            (None, Err(e)) => Err(e),
            (None, Ok(())) => {
                // SAFETY: the batch is done, and the arena is not reused until
                // this job, one of its readers, is released below.
                let outs: Vec<Out<'_>> = self
                    .outs
                    .iter()
                    .zip(&bufs[self.ins.len()..])
                    .map(|(port, &buf)| {
                        let (off, n) = if port.axis == Axis::Rows {
                            (roff, b)
                        } else {
                            (coff, p)
                        };
                        Out {
                            bytes: unsafe {
                                std::slice::from_raw_parts(
                                    buf.add(off * port.row_bytes()),
                                    n * port.row_bytes(),
                                )
                            },
                        }
                    })
                    .collect();
                read(&outs)
            }
        };
        let mut st = self.st.lock().unwrap();
        st.arenas[a].readers -= 1;
        // The last reader frees the arena, and reopens it if nothing is taking
        // reservations (every other one is on the GPU, or still being read).
        if st.arenas[a].readers == 0 {
            st.arenas[a].busy = false;
            if st.open.is_none() {
                st.open_idle();
            }
        }
        self.cv.notify_all();
        res
    }
}

// ── the CUDA runtime, for uploads into the store ───────────────────────────────

/// The few CUDA runtime entry points the store needs, from the `libcudart`
/// ORT already loaded. Resolved at first use; `None` if it cannot be found.
struct Cuda {
    _lib: libloading::Library,
    memcpy_async: unsafe extern "C" fn(*mut u8, *const u8, usize, i32, *mut u8) -> i32,
    stream_create: unsafe extern "C" fn(*mut *mut u8, u32) -> i32,
    stream_sync: unsafe extern "C" fn(*mut u8) -> i32,
}

fn cuda() -> anyhow::Result<&'static Cuda> {
    static CUDA: std::sync::OnceLock<Result<Cuda, String>> = std::sync::OnceLock::new();
    CUDA.get_or_init(|| {
        // SAFETY: loading the CUDA runtime and resolving documented symbols
        // with their documented signatures.
        unsafe {
            let lib = libloading::Library::new("libcudart.so.12").map_err(|e| e.to_string())?;
            let memcpy_async = *lib
                .get::<unsafe extern "C" fn(*mut u8, *const u8, usize, i32, *mut u8) -> i32>(
                    b"cudaMemcpyAsync\0",
                )
                .map_err(|e| e.to_string())?;
            let stream_create = *lib
                .get::<unsafe extern "C" fn(*mut *mut u8, u32) -> i32>(
                    b"cudaStreamCreateWithFlags\0",
                )
                .map_err(|e| e.to_string())?;
            let stream_sync = *lib
                .get::<unsafe extern "C" fn(*mut u8) -> i32>(b"cudaStreamSynchronize\0")
                .map_err(|e| e.to_string())?;
            Ok(Cuda {
                _lib: lib,
                memcpy_async,
                stream_create,
                stream_sync,
            })
        }
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("CUDA runtime unavailable: {e}"))
}

thread_local! {
    /// This thread's non-blocking upload stream: the copies must not
    /// serialize against ORT's streams, and must be waited on by this thread
    /// alone.
    static UPLOAD_STREAM: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Copy `bytes` bytes host -> device and wait for them to land.
///
/// # Safety
/// `dst` must be device memory valid for `bytes`, `src` host memory likewise.
unsafe fn upload(dst: *mut u8, src: *const u8, bytes: usize) -> anyhow::Result<()> {
    let c = cuda()?;
    let stream = UPLOAD_STREAM.with(|s| {
        if s.get() == 0 {
            let mut st: *mut u8 = std::ptr::null_mut();
            // cudaStreamNonBlocking = 1
            let rc = unsafe { (c.stream_create)(&mut st, 1) };
            if rc == 0 {
                s.set(st as usize);
            }
        }
        s.get()
    });
    anyhow::ensure!(stream != 0, "could not create an upload stream");
    // cudaMemcpyHostToDevice = 1
    let rc = unsafe { (c.memcpy_async)(dst, src, bytes, 1, stream as *mut u8) };
    anyhow::ensure!(rc == 0, "cudaMemcpyAsync failed ({rc})");
    let rc = unsafe { (c.stream_sync)(stream as *mut u8) };
    anyhow::ensure!(rc == 0, "cudaStreamSynchronize failed ({rc})");
    Ok(())
}

// ── the h store ───────────────────────────────────────────────────────────────

/// Rows per store chunk, the unit leases are made of: each chunk is one
/// upload, and 64 rows (~300 KB) keeps the copies few and large.
const CHUNK: usize = 64;

/// Trunk features resident on the GPU, indexed by row, for the store-indexed
/// head graphs (`*_g.onnx`, see `python/.../onnx_store.py`).
///
/// A diffusion chain's `h` never changes across its steps, yet the plain graphs
/// took it as an input on every one: at 4.7 KB a row it was ~45% of every
/// diffusion call's bytes. Here a wave uploads each position's row once
/// ([`HStore::lease`]) and every call after passes 8-byte indices.
pub(crate) struct HStore {
    dev: ort::value::Tensor<f16>,
    base: *mut u8,
    stride: usize,
    free: Mutex<Vec<u32>>,
}

// SAFETY: `base` points into `dev`'s device allocation; rows are written only
// through leases, which own them exclusively.
unsafe impl Send for HStore {}
unsafe impl Sync for HStore {}

impl HStore {
    /// A store of `rows` rows of `channels * NUM_CELLS` fp16, on the device
    /// `sess` runs on.
    pub fn new(sess: &Session, channels: usize, rows: usize) -> anyhow::Result<Self> {
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        cuda()?;
        let rows = rows.div_ceil(CHUNK) * CHUNK;
        let alloc = Allocator::new(
            sess,
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(e2a)?,
        )
        .map_err(e2a)?;
        let mut dev =
            ort::value::Tensor::<f16>::new(&alloc, [rows, channels, N, N]).map_err(e2a)?;
        std::mem::forget(alloc);
        let base = dev.data_ptr_mut().cast::<u8>();
        Ok(Self {
            dev,
            base,
            stride: channels * NUM_CELLS,
            free: Mutex::new((0..(rows / CHUNK) as u32).rev().collect()),
        })
    }

    /// The store as a graph input.
    pub fn tensor(&self) -> &ort::value::Tensor<f16> {
        &self.dev
    }

    /// Upload `h` (whole rows) and lease the rows holding it; `None` when the
    /// store is too full, and the caller takes the per-call path instead.
    pub fn lease(&self, h: &[f16]) -> anyhow::Result<Option<Lease<'_>>> {
        let n = h.len() / self.stride;
        let need = n.div_ceil(CHUNK);
        let chunks = {
            let mut free = self.free.lock().unwrap();
            if free.len() < need {
                return Ok(None);
            }
            let keep = free.len() - need;
            free.split_off(keep)
        };
        let lease = Lease {
            store: self,
            chunks,
        };
        for (i, &c) in lease.chunks.iter().enumerate() {
            let rows = (n - i * CHUNK).min(CHUNK);
            let bytes = rows * self.stride * 2;
            // SAFETY: chunk `c` is leased to this call alone, and is `CHUNK`
            // rows of the store; the source is `rows` rows of `h`.
            unsafe {
                upload(
                    self.base.add(c as usize * CHUNK * self.stride * 2),
                    h.as_ptr().add(i * CHUNK * self.stride).cast(),
                    bytes,
                )?;
            }
        }
        Ok(Some(lease))
    }
}

/// Rows of an [`HStore`], held for as long as calls may read them.
pub(crate) struct Lease<'a> {
    store: &'a HStore,
    chunks: Vec<u32>,
}

impl Lease<'_> {
    /// The store row holding leased row `i`.
    pub fn row(&self, i: usize) -> u32 {
        self.chunks[i / CHUNK] * CHUNK as u32 + (i % CHUNK) as u32
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        self.store.free.lock().unwrap().append(&mut self.chunks);
    }
}
