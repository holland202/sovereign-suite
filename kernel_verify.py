#!/usr/bin/env python3
"""
kernel_verify.py — Sovereign Intrinsic Verification Gate
Cryptographically verifies hardware-specific binaries before execution.
"""
import sys, hashlib, os

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"[VERITAS-FATAL] Binary not found: {filepath}")
        sys.exit(1)
    except PermissionError:
        print(f"[VERITAS-FATAL] Permission denied: {filepath}")
        sys.exit(1)

def verify(binary_path, checksum_file="checksums.txt"):
    print(f"[VERITAS-AUTH] Auditing: {binary_path}")
    actual_hash = compute_sha256(binary_path)
    target_basename = os.path.basename(binary_path)
    
    if not os.path.exists(checksum_file):
        print(f"[VERITAS-WARN] No {checksum_file} found. Establishing baseline.")
        with open(checksum_file, "a") as f:
            f.write(f"{actual_hash}  {target_basename}\n")
        print(f"[VERITAS-LOCK] Baseline hash secured: {actual_hash}")
        return True

    with open(checksum_file, "r") as f:
        valid_hashes = f.readlines()

    for line in valid_hashes:
        if target_basename in line:
            expected_hash = line.split()[0].strip()
            if actual_hash == expected_hash:
                print(f"[VERITAS-PASS] Cryptographic lock verified: {actual_hash[:8]}...")
                return True
            else:
                print(f"[VERITAS-FATAL] TOPOLOGY CORRUPTION DETECTED in {target_basename}")
                print(f"Expected: {expected_hash}")
                print(f"Actual:   {actual_hash}")
                sys.exit(1)
    
    print(f"[VERITAS-WARN] Binary {target_basename} not in {checksum_file}. Appending to baseline.")
    with open(checksum_file, "a") as f:
        f.write(f"{actual_hash}  {target_basename}\n")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 kernel_verify.py <path_to_binary> [checksum_file]")
        sys.exit(1)
    
    target = sys.argv[1]
    db = sys.argv[2] if len(sys.argv) > 2 else "checksums.txt"
    verify(target, db)
