import os
from pathlib import Path
from dotenv import load_dotenv
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv(ROOT_DIR / ".env")

class MultiTenantConfig:
    """Multi-tenant clinical application configuration."""
    # Tenant Details
    TENANT_ID: str = os.getenv("TENANT_ID", "CL_UCH_001")
    TENANT_NAME: str = os.getenv("TENANT_NAME", "UC Health")
    TENANT_SLUG: str = os.getenv("TENANT_SLUG", "uc_health")
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    OPENAI_FALLBACK_MODEL: str = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")

    # Snowflake Credentials
    SNOWFLAKE_USER: str = os.getenv("SNOWFLAKE_USER", "SA_MS_Lambda")
    SNOWFLAKE_ACCOUNT: str = os.getenv("SNOWFLAKE_ACCOUNT", "NBNAETU-CS50192")
    SNOWFLAKE_ROLE: str = os.getenv("SNOWFLAKE_ROLE", "DB_SUPPLYCOPIA_TRANSFORM_ETL")
    SNOWFLAKE_WAREHOUSE: str = os.getenv("SNOWFLAKE_WAREHOUSE", "DATAMANAGEMENT_WH")
    SNOWFLAKE_DATABASE: str = os.getenv("SNOWFLAKE_DATABASE", "SUPPLYCOPIA_TRANSFORM")
    SNOWFLAKE_SCHEMA: str = os.getenv("SNOWFLAKE_SCHEMA", "SCRETAIL_IMPORT")
    
    # Key Authentication
    PRIVATE_KEY_PATH: str = os.getenv("PRIVATE_KEY_PATH", str(ROOT_DIR / "lambda_migration._rsa_key.p8"))
    PRIVATE_KEY_PASSPHRASE: str = os.getenv("PRIVATE_KEY_PASSPHRASE", "")

    # Source Multi-Tenant Table
    SOURCE_TABLE: str = os.getenv("SOURCE_TABLE", "SC_MULTI_TENANT_CONSUMPTION_SAVINGS")

    # Dynamic Tenant Table Names
    @classmethod
    def get_prefix(cls) -> str:
        return cls.TENANT_SLUG.upper()

    @classmethod
    def anomaly_table(cls) -> str:
        return f"{cls.get_prefix()}_HYBRID_ANOMALY_DETECTION"

    @classmethod
    def bom_table(cls) -> str:
        return f"{cls.get_prefix()}_STANDARD_BOMS_REPORT"

    @classmethod
    def master_matrix_table(cls) -> str:
        return f"{cls.get_prefix()}_MASTER_SUBSTITUTION_MATRIX"

    @classmethod
    def physician_matrix_table(cls) -> str:
        return f"{cls.get_prefix()}_PHYSICIAN_SUBSTITUTION_MATRIX"

    # Local Artifact Paths
    @classmethod
    def raw_data_path(cls) -> Path:
        return DATA_DIR / f"{cls.TENANT_SLUG}_utilization_data.csv"

    @classmethod
    def anomaly_data_path(cls) -> Path:
        return DATA_DIR / f"{cls.TENANT_SLUG}_hybrid_anomaly.csv"

    @classmethod
    def bom_data_path(cls) -> Path:
        return DATA_DIR / f"{cls.TENANT_SLUG}_standard_boms.csv"

    @classmethod
    def master_matrix_path(cls) -> Path:
        return DATA_DIR / f"{cls.TENANT_SLUG}_master_matrix.csv"

    @classmethod
    def physician_matrix_path(cls) -> Path:
        return DATA_DIR / f"{cls.TENANT_SLUG}_physician_matrix.csv"

    @classmethod
    def rlhf_path(cls) -> Path:
        return DATA_DIR / "rlhf_preference_cards.jsonl"


def get_snowflake_connection():
    """Establishes an authenticated Snowflake connection using RSA private key."""
    key_path = Path(MultiTenantConfig.PRIVATE_KEY_PATH)
    if not key_path.exists():
        fallback = ROOT_DIR / "lambda_migration._rsa_key.p8"
        if fallback.exists():
            key_path = fallback
        else:
            raise FileNotFoundError(f"Snowflake RSA private key not found at {key_path}")

    with open(key_path, "rb") as key_file:
        key_data = key_file.read()

    passphrase = MultiTenantConfig.PRIVATE_KEY_PASSPHRASE.strip()
    pass_bytes = passphrase.encode() if passphrase and passphrase.lower() != "none" else None

    try:
        p_key = serialization.load_pem_private_key(
            key_data,
            password=pass_bytes,
            backend=default_backend()
        )
    except ValueError as e:
        if "Password was given but private key is not encrypted" in str(e):
            p_key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )
        else:
            raise e

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    return snowflake.connector.connect(
        user=MultiTenantConfig.SNOWFLAKE_USER,
        account=MultiTenantConfig.SNOWFLAKE_ACCOUNT,
        private_key=pkb,
        role=MultiTenantConfig.SNOWFLAKE_ROLE,
        warehouse=MultiTenantConfig.SNOWFLAKE_WAREHOUSE,
        database=MultiTenantConfig.SNOWFLAKE_DATABASE,
        schema=MultiTenantConfig.SNOWFLAKE_SCHEMA
    )
