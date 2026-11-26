# Compiler weights, version 1

`trillim.tdl.core` defines logical dtypes; `trillim.compiler.dtype` implements their canonical encodings. Its encoder and decoder registries contain a function for each dtype. Native integer values retain their NumPy dtype through encoding and decoding, including INT64 and UINT64. BF16 and quantized formats decode to FP32. Backend-specific layouts belong to the compiler.

The compiler uses these private weight helpers:

- `_resolve_weights(graph, checkpoints) -> _WeightIndex`
- `_resolve_payloads(index, requests=None) -> Iterator[_ResolvedWeight]`
- `_write_weights(weights, destination) -> _WeightManifest`
- `_read_manifest(bundle) -> _WeightManifest`

`reader.py` reads checkpoint metadata and raw byte slices. Configuration and safetensors metadata remain available without interpreting them. `resolve.py` owns aliases, source-format interpretation and every conversion to the requested Trillim dtype and packing. It supplies output records and bounded iterators of final encoded byte chunks. `bundle.py` assigns file offsets, writes those chunks and publishes the completed bundle; it does not decode or encode tensors.

`checkpoints` maps TDL model names to a safetensors file, shard index, or checkpoint directory. Resolution validates tensor extents and indexed shard membership, then binds concrete shapes and logical dtypes to weight constants. Identities include the checkpoint name, so equal tensor names in different checkpoints remain distinct.

A TDL weight's explicit dtype takes precedence over its constant's dtype. Otherwise, resolution preserves the checkpoint tensor's logical dtype, including imported quantized formats.

Tied tensors declared by checkpoint metadata or `tie_word_embeddings` share an identity. If both copies are stored, their shapes, dtypes and source bytes must agree. Each requested dtype or packing is a separate representation. Requests for the same identity, dtype and packing share one payload span.

## Binary layout

The bundle is one file:

1. An eight-byte little-endian unsigned JSON-header length.
2. A UTF-8 JSON header, padded with spaces to a 64-byte boundary.
3. Tensor payloads in their selected physical layouts, each beginning at a 64-byte boundary.

The header length includes its trailing spaces. `data_start` is the absolute start of the payload area. Each entry's `offset` is relative to that area, so its file address is `data_start + offset`. All multibyte values are little-endian.

The header contains `format="trillim.weights"`, `version=1`, `endian="little"`, `data_start`, and an `entries` mapping. Entries record their name, logical reference, dtype, shape, layout, byte extent, representation aliases, import method, and source provenance. Provenance includes the checkpoint-qualified tensor name, original file extent, source dtype/shape, and SHA256 of the source tensor bytes.

`canonical` payloads store value bytes first, followed by their scale sections. Grouped INT5 adds high bits after its BF16 scales. Grouped INT6 then adds high bits and signed subscales. Matrix ternary stores one FP32 gamma after its values. MXFP4 stores E8M0 scales after its packed values. `trillim.compiler.dtype._payload_sections` returns the section names and byte lengths. Internal sections have no alignment padding; readers must handle unaligned scales.

`row_int8_block32` retains logical dtype `row_int8`, dimensions `[outputs, inputs]`, and the same payload size. Codes are stored as `[outputs/32, inputs/32, 32, 32]`, followed by the unchanged BF16 scale for each output row. Both dimensions must be positive multiples of 32. The compiler selects this layout for compatible kernels; TDL authors continue to specify the logical dtype. Benchmarks repack the same encoded codes and scales when comparing physical layouts, rather than requantizing them.

Resolve supplies each encoded chunk with its offset within the tensor payload, keeping values, scales and other sections separate. The writer places those bytes at the corresponding file positions. It writes a temporary file and replaces the destination after completion.

## Checkpoint imports

Native tensors copy directly when the requested dtype matches the source. This preserves native integer values and BF16 bits. Other requested representations convert the effective logical values.

BitNet projection tensors are imported using their pretrained semantics. Floating latent weights use global absmean quantization with the source floating precision applied to the effective gamma. They must not pass through the generic `mat_ternary` absmax quantizer. Offline BitNet imports preserve ternary codes and interpret the scalar scale according to the declared `bitlinear` or `autobitlinear` convention. Further dtype overrides convert these effective weights.

MXFP4 `_blocks` and `_scales` imports preserve checkpoint codes and E8M0 scales when the requested dtype remains MXFP4. Conversion to another dtype decodes those blocks in bounded chunks.

FP8 `F8_E4M3` projections with `weight_scale_inv` and `quantization_config.weight_block_size=[128,128]` import as `fp8_e4m3`. The payload retains row-major E4M3FN codes followed by the row-major FP32 scale grid for 128-by-128 tiles. Source FP16/BF16 scales widen exactly to FP32. Native import preserves codes without requantization; dtype overrides decode scaled values in bounded chunks. Tied FP8 copies must agree in both codes and effective scales, and provenance records both source tensors.

Safetensors headers and shard indices are validated before binding. Duplicate names, invalid extents, overlapping data, missing shards and conflicting declared ties fail compilation. Source size or modification-time changes during conversion also fail compilation.
