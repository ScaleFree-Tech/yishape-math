/**
 * YSGP binary protocol for GPU/HPC graph execution.
 *
 * <h2>YSGP Binary Protocol Specification v2</h2>
 *
 * <p>The YSGP (YiShape Graph Protocol) is a little-endian binary format for
 * transmitting autodiff computation graphs from Java to Rust backends (GPU/HPC).
 * It replaces JSON serialization for graphs with &gt;500 nodes, reducing overhead
 * by 10-100× and eliminating floating-point serialization artifacts.</p>
 *
 * <h3>Header</h3>
 * <pre>
 *   u32 magic  = 0x50535359 ("YSGP")
 *   u32 version  = 1 or 2
 *   u32 num_nodes
 * </pre>
 *
 * <h3>Per-node layout</h3>
 * <pre>
 *   u16 flags        — bitmask (see flag definitions)
 *   u16 op_len       — UTF-8 byte length of op tag
 *   u32 id           — node's position in the topological order
 *   u8[op_len] op    — op tag string (UTF-8, not null-terminated)
 *   u16 num_dims
 *   u32[num_dims] shape  — dimensions of this node's output tensor
 *   u16 num_inputs
 *   u32[num_inputs] inputs — node IDs of input tensors
 *   [if FLAG_HAS_INPUT_SHAPES]:
 *     u16 num_in_shapes
 *     for each input:
 *       u16 ndim
 *       u32[ndim] input_shape
 *   [if FLAG_HAS_SCALAR]:   f64 scalar_param
 *   [if FLAG_HAS_PARAM2]:   f64 scalar_param2
 *   [if FLAG_HAS_DATA]:     u32 data_len, f64[data_len]
 *   [if FLAG_HAS_INDICES]:  u32 indices_len, i32[indices_len]
 *   [VERSION &ge; 2 only]:  u16 extension_block_len, u8[extension_block_len]
 * </pre>
 *
 * <h3>Flag bits</h3>
 * <table>
 *   <tr><th>Bit</th><th>Value</th><th>Name</th><th>Data</th></tr>
 *   <tr><td>0</td><td>1</td><td>FLAG_HAS_DATA</td><td>u32 len + f64[len]</td></tr>
 *   <tr><td>1</td><td>2</td><td>FLAG_HAS_SCALAR</td><td>f64 (8 bytes)</td></tr>
 *   <tr><td>2</td><td>4</td><td>FLAG_HAS_PARAM2</td><td>f64 (8 bytes)</td></tr>
 *   <tr><td>3</td><td>8</td><td>FLAG_HAS_INDICES</td><td>u32 len + i32[len]</td></tr>
 *   <tr><td>4</td><td>16</td><td>FLAG_IS_LEAF</td><td>(none)</td></tr>
 *   <tr><td>5</td><td>32</td><td>FLAG_HAS_INPUT_SHAPES</td><td>(V2+ only) u16 count + per-input shapes for broadcast mode</td></tr>
 * </table>
 *
 * <h3>Result format (Rust &rarr; Java)</h3>
 * <pre>
 *   f64 loss
 *   u32 num_grads
 *   for each grad:
 *     u32 grad_len
 *     f64[grad_len] grad_data
 * </pre>
 *
 * <h3>Version compatibility</h3>
 * <p><b>V1</b> (default wire version): Fixed flag layout. The Rust parser must
 * handle exactly the flags listed above. Unknown flags cause parse errors on the
 * Rust side (e.g., "unexpected EOF").</p>
 *
 * <p><b>V2</b> (opt-in via {@code -Dyishape.ysgp.version=2}): Adds a
 * {@code u16 extension_block_len} trailer per node. The extension block contains
 * zero or more TLV entries: {@code u16 type_tag, u16 data_len, u8[data_len]}.
 * Readers skip unknown extension tags by advancing {@code data_len} bytes.
 * New flag bits should be added as extension TLV entries rather than raw flag bits,
 * ensuring old Rust binaries can skip them safely.</p>
 *
 * <h3>Adding a new flag</h3>
 * <ol>
 *   <li>Define the flag constant in {@code TensorBinaryProtocol}</li>
 *   <li>Add a {@code FlagMeta} entry to {@code FLAG_METADATA}</li>
 *   <li>Update {@code nodeSize()} and {@code writeNode()} for the new data</li>
 *   <li>Update both Rust parsers (GPU and HPC crates)</li>
 *   <li>Add protocol tests in {@code ProtocolContractTest}</li>
 *   <li>Bump the wire version if the change is backward-incompatible</li>
 * </ol>
 *
 * @see com.yishape.lab.math.autodiff.graph.binary.BinaryProtocol
 * @see com.yishape.lab.math.autodiff.graph.binary.TensorBinaryProtocol
 */
package com.yishape.lab.math.autodiff.graph.binary;
