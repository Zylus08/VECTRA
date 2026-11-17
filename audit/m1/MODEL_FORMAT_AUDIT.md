# Model Format Audit: GGUF Implementation & Inspector

## Specification Conformance
The VECTRA GGUF parser is an original, read-only implementation written to the official GGUF v2/v3 binary specification without incorporating code from GGML or llama.cpp.

### Implemented & Tested Format Features
1. **Magic Number Verification**: Explicit 4-byte check for `0x46554747` (`GGUF` in little-endian order).
2. **Version Negotiation**: Validates version number; accepts versions 2 and 3. Throws descriptive error on unknown versions.
3. **Metadata KV Decoding**:
   - Supported scalar types: `UINT8`, `INT8`, `UINT16`, `INT16`, `UINT32`, `INT32`, `FLOAT32`, `BOOL`, `STRING`, `UINT64`, `INT64`, `FLOAT64`.
   - Supported array types: `STRING[]`, `INT32[]`, `FLOAT32[]`.
   - Duplicate key detection: Throws an exception if duplicate keys are detected in the metadata header.
4. **Data Alignment Handling**: Reads `general.alignment` (defaults to 32 bytes) and advances file offset via aligned padding.
5. **Tensor Table Verification**:
   - Parses tensor names, dimension counts (1D to 4D), dimension sizes, tensor types, and relative offsets.
   - Computes absolute file offset (`data_offset + tensor_offset`).
   - Bounds validation: Verifies that every tensor's declared footprint fits strictly within the buffer boundaries.
6. **Endianness Check**: Explicit host architecture assertion ensuring little-endian byte ordering.

## Negative Test Audit
The parser was subjected to negative test vectors in `tests/unit/unit_tests.cpp`:
- `bad_magic.gguf`: Contains `0x12345678`. The parser successfully detected the corrupted magic and threw `VectraException`.
- `corrupt_offset.gguf`: Contains a valid header but declares a tensor offset exceeding physical file size. The parser threw `VectraException("Tensor bad_tensor data extends past end of file")`.

## CLI Inspector Tool (`vectra_inspect`)
The standalone utility `vectra_inspect` allows inspecting GGUF files without executing inference:
- Parses file header and architecture metadata.
- Outputs table of tensors, dimensions, data types, and estimated byte sizes.
- Verified on `tests/fixtures/valid_test.gguf`.
