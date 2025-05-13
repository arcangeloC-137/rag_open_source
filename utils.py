import hashlib

def compute_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()