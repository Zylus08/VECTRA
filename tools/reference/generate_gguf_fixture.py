import struct
import os
import sys

def write_string(f, s):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)

def create_valid_gguf(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        # Magic 'GGUF'
        f.write(struct.pack('<I', 0x46554747))
        # Version 3
        f.write(struct.pack('<I', 3))
        # Tensor count: 2
        f.write(struct.pack('<Q', 2))
        # Metadata KV count: 5
        f.write(struct.pack('<Q', 5))
        
        # KV 1: general.architecture (STRING = 8)
        write_string(f, "general.architecture")
        f.write(struct.pack('<I', 8))
        write_string(f, "qwen2")
        
        # KV 2: general.alignment (UINT32 = 4)
        write_string(f, "general.alignment")
        f.write(struct.pack('<I', 4))
        f.write(struct.pack('<I', 32))
        
        # KV 3: qwen2.context_length (UINT32 = 4)
        write_string(f, "qwen2.context_length")
        f.write(struct.pack('<I', 4))
        f.write(struct.pack('<I', 2048))
        
        # KV 4: qwen2.embedding_length (UINT32 = 4)
        write_string(f, "qwen2.embedding_length")
        f.write(struct.pack('<I', 4))
        f.write(struct.pack('<I', 64))
        
        # KV 5: qwen2.attention.layer_norm_rms_epsilon (FLOAT32 = 6)
        write_string(f, "qwen2.attention.layer_norm_rms_epsilon")
        f.write(struct.pack('<I', 6))
        f.write(struct.pack('<f', 1e-6))
        
        # Tensor 1: token_embd.weight
        write_string(f, "token_embd.weight")
        f.write(struct.pack('<I', 2)) # 2 dims
        f.write(struct.pack('<Q', 64))
        f.write(struct.pack('<Q', 16))
        f.write(struct.pack('<I', 0)) # F32 = 0
        f.write(struct.pack('<Q', 0)) # offset 0
        
        # Tensor 2: blk.0.attn_norm.weight
        write_string(f, "blk.0.attn_norm.weight")
        f.write(struct.pack('<I', 1)) # 1 dim
        f.write(struct.pack('<Q', 64))
        f.write(struct.pack('<I', 0)) # F32 = 0
        t1_bytes = 64 * 16 * 4 # 4096 bytes
        # Alignment is 32, so 4096 is already aligned
        f.write(struct.pack('<Q', t1_bytes))
        
        # Align header to 32 bytes before tensor data
        cur = f.tell()
        rem = cur % 32
        if rem != 0:
            f.write(b'\x00' * (32 - rem))
            
        # Tensor 1 data: 64 * 16 * 4 bytes
        f.write(b'\x00' * (64 * 16 * 4))
        # Tensor 2 data: 64 * 4 bytes
        f.write(b'\x00' * (64 * 4))
    print(f"Created valid GGUF fixture at {path}")

def create_invalid_magic_gguf(path):
    with open(path, "wb") as f:
        f.write(struct.pack('<I', 0x12345678)) # Bad magic
        f.write(struct.pack('<I', 3))
        f.write(b'\x00' * 32)
    print(f"Created invalid magic GGUF fixture at {path}")

def create_corrupt_offset_gguf(path):
    with open(path, "wb") as f:
        f.write(struct.pack('<I', 0x46554747)) # Magic
        f.write(struct.pack('<I', 3))          # Version
        f.write(struct.pack('<Q', 1))          # 1 tensor
        f.write(struct.pack('<Q', 0))          # 0 metadata
        write_string(f, "bad_tensor")
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<Q', 100))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<Q', 999999999))  # Out of bounds offset
    print(f"Created corrupt offset GGUF fixture at {path}")

if __name__ == "__main__":
    fixtures_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
    os.makedirs(fixtures_dir, exist_ok=True)
    create_valid_gguf(os.path.join(fixtures_dir, "valid_test.gguf"))
    create_invalid_magic_gguf(os.path.join(fixtures_dir, "bad_magic.gguf"))
    create_corrupt_offset_gguf(os.path.join(fixtures_dir, "corrupt_offset.gguf"))
