import os

# Must run before `app.config` is imported anywhere.
# Tests use the public dev tenant keys, so dev mode is enabled explicitly.
os.environ["AEGIS_DEV_MODE"] = "true"
os.environ["ADMIN_API_KEY"] = "test-admin-key-not-for-production-0123456789"
