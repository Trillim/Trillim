# Compiler weights, version 1

`trillim.tdl.core` defines logical dtypes; `trillim.compiler.dtype` implements their canonical encodings. Its encoder and decoder registries contain a function for each dtype. Native integer values retain their NumPy dtype through encoding and decoding, including INT64 and UINT64. BF16 and quantized formats decode to FP32. Backend-specific layouts belong to the compiler.

The compiler weight API is:

- `resolve_weights(graph, checkpoints) -> WeightIndex`
- `write_weights(index, destination, required_representations=None) -> WeightManifest`
- `read_manifest(bundle) -> WeightManifest`

`checkpoints` maps TDL model names to a safetensors file, shard index, or checkpoint directory. Resolution validates tensor extents and indexed shard membership, then binds concrete shapes and logical dtypes to weight constants. Identities include the checkpoint name, so equal tensor names in different checkpoints remain distinct.

A TDL weight's explicit dtype takes precedence over its constant's dtype. Otherwise, resolution preserves the checkpoint tensor's logical dtype, including imported quantized formats.

Tied tensors declared by checkpoint metadata or `tie_word_embeddings` share an identity. If both copies are stored, their shapes, dtypes and source bytes must agree. Each requested dtype or packing is a separate representation. Requests for the same identity, dtype and packing share one payload span.

## Binary layout

The bundle is one file:

1. An eight-byte little-endian unsigned JSON-header length.
2. A UTF-8 JSON header, padded with spaces to a 64-byte boundary.
3. Canonical tensor payloads, each beginning at a 64-byte boundary.

The header length includes its trailing spaces. `data_start` is the absolute start of the payload area. Each entry's `offset` is relative to that area, so its file address is `data_start + offset`. All multibyte values are little-endian.

The header contains `format="trillim.weights"`, `version=1`, `endian="little"`, `data_start`, and an `entries` mapping. Entries record their name, logical reference, dtype, shape, layout, byte extent, representation aliases, import method, and source provenance. Provenance includes the checkpoint-qualified tensor name, original file extent, source dtype/shape, and SHA256 of the source tensor bytes.

`canonical` payloads store value bytes first, followed by their scale sections. Grouped INT5 adds high bits after its BF16 scales. Grouped INT6 then adds high bits and signed subscales. Matrix ternary stores one FP32 gamma after its values. MXFP4 stores E8M0 scales after its packed values. `trillim.compiler.dtype.payload_sections` returns the section names and byte lengths. Internal sections have no alignment padding; readers must handle unaligned scales.

The writer streams bounded row or column chunks into each section's final position. It does not concatenate independently encoded chunks, which would interleave value and scale sections incorrectly. It writes a temporary file and replaces the destination after completion.

## Checkpoint imports

Native tensors copy directly when the requested dtype matches the source. This preserves native integer values and BF16 bits. Other requested representations convert the effective logical values.

BitNet projection tensors are imported using their pretrained semantics. Floating latent weights use global absmean quantization with the source floating precision applied to the effective gamma. They must not pass through the generic `mat_ternary` absmax quantizer. Offline BitNet imports preserve ternary codes and interpret the scalar scale according to the declared `bitlinear` or `autobitlinear` convention. Further dtype overrides convert these effective weights.

MXFP4 `_blocks` and `_scales` imports preserve checkpoint codes and E8M0 scales when the requested dtype remains MXFP4. Conversion to another dtype decodes those blocks in bounded chunks.

Safetensors headers and shard indices are validated before binding. Duplicate names, invalid extents, overlapping data, missing shards and conflicting declared ties fail compilation. Source size or modification-time changes during conversion also fail compilation.
