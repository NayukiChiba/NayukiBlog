import hashlib

def hash_password(password: str) -> str:
    """
    Hashes a password using SHA-256.
    This is a one-way function: you cannot decrypt the password.
    To verify, you hash the input and compare it with the stored hash.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain password against a hashed password.
    """
    return hash_password(plain_password) == hashed_password
