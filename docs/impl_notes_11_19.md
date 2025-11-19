# implmentation notes (11/19)

## summary

currently only `256` and `4096` is implemented.
These are global memory bound, and both HadaCore and ours achieve the same performance as a in-place add kernel.

We note that `Hadacore` is only significantly slower than `add one` on sizes `16384` and `32768`.
We believe it is possible to achieve the same performance as `add one` on these too.

```
test_sizes_m:  [256, 4096]
test_elem_counts:  [134217728, 268435456]
cases:  ['AddOne', 'HadaCore', 'Ours']
running on NVIDIA A100-SXM4-80GB

Testing size 256x524288
AddOne m=256 n=524288 float16 | 0.3202 ms/iter over 200 iters
HadaCore m=256 n=524288 float16 | 0.3203 ms/iter over 200 iters
Ours m=256 n=524288 float16 | 0.3242 ms/iter over 200 iters
Testing size 256x1048576
AddOne m=256 n=1048576 float16 | 0.6362 ms/iter over 200 iters
HadaCore m=256 n=1048576 float16 | 0.6355 ms/iter over 200 iters
Ours m=256 n=1048576 float16 | 0.6428 ms/iter over 200 iters
Testing size 4096x32768
AddOne m=4096 n=32768 float16 | 0.3204 ms/iter over 200 iters
HadaCore m=4096 n=32768 float16 | 0.3250 ms/iter over 200 iters
Ours m=4096 n=32768 float16 | 0.3224 ms/iter over 200 iters
Testing size 4096x65536
AddOne m=4096 n=65536 float16 | 0.6361 ms/iter over 200 iters
HadaCore m=4096 n=65536 float16 | 0.6451 ms/iter over 200 iters
Ours m=4096 n=65536 float16 | 0.6389 ms/iter over 200 iters
4/4 size tests done
Wrote benchmark_2025-11-19_16-51-02.json
```

## Key Progress

We created abstraction for handling hadamard transforms, significantly simplifying code. During this we also find that the transform can be done with no `__shfl_sync` instructions.

HadaCore: Hadamard butterflies across lane partitions were implemented with warp shuffles and bespoke swizzles. This was correct but brittle and verbose; every change in where data “lived” required more swizzle code.

Ours: Treat the swizzle as metadata. Logical axes are tracked by a permutation object P attached to the data. Functions compose these permutations instead of moving data. When moving to/from memory, we materialize the permutation by pointer arithmetic; inside a warp, we never shuffle. The cross-lane mixing needed for the Hadamard is handled by a single mma.sync, not by shuffles.

### size 256 Hadamard, by HadaCore

```asm
ld.shared.u32 	%r167, [%r120+512];
ld.shared.u32 	%r168, [%r132+512];
ld.shared.u32 	%r169, [%r144+768];
ld.shared.u32 	%r170, [%r156+768];
selp.b32 	%r47, %r167, %r168, %p5;
selp.b32 	%r171, %r168, %r167, %p5;
selp.b32 	%r57, %r169, %r170, %p5;
selp.b32 	%r172, %r170, %r169, %p5;
shfl.sync.bfly.b32 	%r48|%p12, %r171, %r107, %r86, %r160;
shfl.sync.bfly.b32 	%r58|%p13, %r172, %r107, %r86, %r160;
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r41, %r42}, {%r75, %r75, %r75, %r76}, {%r47, %r48}, {%r80, %r80};
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r51, %r52}, {%r75, %r75, %r75, %r76}, {%r57, %r58}, {%r80, %r80};
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r61, %r62}, {%r75, %r75, %r75, %r76}, {%r41, %r51}, {%r80, %r80};
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r71, %r72}, {%r75, %r75, %r75, %r76}, {%r42, %r52}, {%r80, %r80};
shfl.sync.bfly.b32 	%r173|%p14, %r71, %r107, %r86, %r160;
shfl.sync.bfly.b32 	%r174|%p15, %r72, %r107, %r86, %r160;
selp.b32 	%r175, %r61, %r173, %p5;
selp.b32 	%r176, %r173, %r61, %p5;
selp.b32 	%r177, %r62, %r174, %p5;
selp.b32 	%r178, %r174, %r62, %p5;
st.global.u32 	[%rd16+512], %r175;
st.global.u32 	[%rd18+512], %r176;
st.global.u32 	[%rd20+512], %r177;
st.global.u32 	[%rd22+512], %r178;
```

### size 256 Hadamard, by us

```asm
ld.global.u32 	%r127, [%rd6+1536];
ld.global.u32 	%r128, [%rd6+1664];
ld.global.u32 	%r137, [%rd6+1792];
ld.global.u32 	%r138, [%rd6+1920];
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r121, %r122}, {%r155, %r155, %r155, %r156}, {%r127, %r128}, {%r160, %r160};
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r131, %r132}, {%r155, %r155, %r155, %r156}, {%r137, %r138}, {%r160, %r160};
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r141, %r142}, {%r155, %r155, %r155, %r156}, {%r121, %r131}, {%r160, %r160};
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%r151, %r152}, {%r155, %r155, %r155, %r156}, {%r122, %r132}, {%r160, %r160};
st.global.u32 	[%rd7+1536], %r141;
st.global.u32 	[%rd7+1664], %r151;
st.global.u32 	[%rd7+1792], %r142;
st.global.u32 	[%rd7+1920], %r152;
```

### Core ideas and building blocks

1. Perm: algebra for axis bookkeeping

- A small struct encodes a permutation for N axes:
  ```
  template <int N> struct Perm { array<int, N> ord; ... };
  ```
- You never physically permute data to change layouts. You only change the permutation you attach to a fragment. This eliminates swizzle code.

2. Frag: “nd-array view” of the warp tile

- A fragment is the contents of a 2x…x2 tile as seen by one lane:
  ```c++
  template <DType dtype, int N, Perm<N> P = Perm<N>::ID()>
  struct Frag { array<u16, 1 << (N - 5)> data; ... };
  ```

3. rotate_4: a single-MMA 4-axis Hadamard

- Spec: apply a 4-axis Hadamard on logical axes `{0,1,5,6}` of a 7D tile, then applying permutation `{3,4,5,0,1,2,6}`
- Permutation is implied by swizzle needed by Nvidia.

4. 8-axis Hadamard from two rotate_4 calls

- A full 8D Hadamard (size 256 per lane) is built from two 4-axis steps and a local transpose
