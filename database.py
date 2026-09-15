"""
database.py - MongoDB connection and collection helpers for AegisAI.

Provides a singleton MongoClient and exposes all collections used
across the application.  Import `db` and use `db.<collection>` directly.

Collections:
  - users           : registered user accounts
  - sessions        : active auth session tokens → email mapping
  - suppliers       : supplier catalog per workspace
  - purchase_orders : PO documents per workspace
  - customers       : customer registry per workspace
  - onboarding      : business onboarding config per workspace
  - whatsapp_logs   : outgoing / incoming WhatsApp message log
"""

import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import config

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_client: MongoClient = None
_db = None


def get_db():
    """Return the MongoDB database handle, connecting on first call."""
    global _client, _db
    if _db is not None:
        return _db

    uri = config.MONGO_URI
    if not uri:
        print(
            "[database] WARNING: MONGO_URI is not set. "
            "All DB operations will fail. "
            "Set MONGO_URI in your .env file.",
            file=sys.stderr,
        )
        return None

    try:
        _client = MongoClient(
            uri,
            serverSelectionTimeoutMS=15000,  # 15-second timeout
            connectTimeoutMS=15000,
            socketTimeoutMS=15000,
            appName="aegisai",
            retryWrites=True,
            retryReads=True,
        )
        # Force a connection attempt to validate the URI early
        _client.admin.command("ping")
        _db = _client["aegisai"]
        print("[database] [OK] Connected to MongoDB Atlas (aegisai database).")
        _create_indexes(_db)
        return _db
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        print(f"[database] [ERROR] Could not connect to MongoDB: {exc}", file=sys.stderr)
        print(
            "[database] FIX: Go to https://cloud.mongodb.com -> Network Access "
            "-> Add IP Address -> Add 0.0.0.0/0 (Allow from anywhere)",
            file=sys.stderr,
        )
        _client = None
        _db = None
        return None


def _create_indexes(database):
    """Create useful indexes on startup (idempotent)."""
    try:
        database["users"].create_index("email", unique=True, background=True)
        database["sessions"].create_index("token", unique=True, background=True)
        database["sessions"].create_index("email", background=True)
        database["suppliers"].create_index(
            [("workspace_id", 1), ("name", 1)], background=True
        )
        database["purchase_orders"].create_index(
            [("workspace_id", 1), ("po_id", 1)], unique=True, background=True
        )
        database["customers"].create_index(
            [("workspace_id", 1), ("email", 1)], background=True
        )
        database["onboarding"].create_index("workspace_id", unique=True, background=True)
        database["whatsapp_logs"].create_index("workspace_id", background=True)
    except Exception as exc:
        print(f"[database] Index creation warning: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_collection(name: str):
    """Return a named collection from the aegisai database."""
    database = get_db()
    if database is None:
        raise RuntimeError(
            "Database connection failed. "
            "Set MONGO_URI in environment variables and whitelist 0.0.0.0/0 in MongoDB Atlas."
        )
    return database[name]


# Convenience accessors (lazily initialized)
class _Collections:
    """Lazy-proxy for MongoDB collections."""

    def __getattr__(self, name: str):
        return get_collection(name)


db = _Collections()
