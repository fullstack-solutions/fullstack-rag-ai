import os
import hashlib
import pickle
from typing import Any, Tuple, Optional


class FileUtils:
    """
    Utility class for file hashing and safe binary load/save operations.
    """

    # -----------------------------
    # File Hash
    # -----------------------------
    @staticmethod
    def get_file_hash(file_path: str) -> Tuple[bool, Optional[str], str]:
        """
        Compute the MD5 hash of a file.
        """
        try:
            hash_md5 = hashlib.md5()

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

            return True, hash_md5.hexdigest(), "Hash generated successfully"

        except FileNotFoundError:
            return False, None, "File not found"

        except PermissionError:
            return False, None, "Permission denied"

        except Exception as e:
            return False, None, f"[get_file_hash] {e}"

    # -----------------------------
    # Load Binary
    # -----------------------------
    @staticmethod
    def load_binary(path: str) -> Tuple[bool, Any, str]:
        """
        Load a Python object from a pickle file safely.
        """
        try:
            if not os.path.exists(path):
                return False, None, "File does not exist"

            with open(path, "rb") as f:
                return True, pickle.load(f), "Load successful"

        except pickle.UnpicklingError:
            return False, None, "Corrupted pickle file"

        except Exception as e:
            return False, None, f"[load_binary] {e}"

    # -----------------------------
    # Save Binary
    # -----------------------------
    @staticmethod
    def save_binary(path: str, data: Any) -> Tuple[bool, str]:
        """
        Save a Python object to a pickle file safely.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(data, f)

            return True, "Save successful"

        except PermissionError:
            return False, "Permission denied"

        except Exception as e:
            return False, f"[save_binary] {e}"