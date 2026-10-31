import os
import sys
import time
import json
import uuid
import zlib
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
import xml.etree.ElementTree as ET
from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse, Response

BASE_DIR = Path(__file__).resolve().parent.parent

# Load SSO Configuration
IDP_ENTITY_ID = os.getenv("SAML_IDP_ENTITY_ID", "https://sts.windows.net/ae81ad54-d58a-4097-9fc7-98a2954a7a6d/")
IDP_SSO_URL = os.getenv("SAML_IDP_SSO_URL", "https://login.microsoftonline.com/ae81ad54-d58a-4097-9fc7-98a2954a7a6d/saml2")
IDP_SLO_URL = os.getenv("SAML_IDP_SLO_URL", "https://login.microsoftonline.com/ae81ad54-d58a-4097-9fc7-98a2954a7a6d/saml2")
IDP_METADATA_URL = os.getenv("SAML_IDP_METADATA_URL", "https://login.microsoftonline.com/ae81ad54-d58a-4097-9fc7-98a2954a7a6d/federationmetadata/2007-06/federationmetadata.xml?appid=86829424-5c87-4248-9a32-18b3702137dc")

SP_ENTITY_ID = os.getenv("SAML_SP_ENTITY_ID", "https://uchealth-clinical.supplycopia.com")
SP_ACS_URL = os.getenv("SAML_SP_ACS_URL", "https://uchealth-clinical.supplycopia.com/saml/acs")
SP_SLO_URL = os.getenv("SAML_SP_SLO_URL", "https://uchealth-clinical.supplycopia.com/saml/sls")

CERT_PATH = os.getenv("SAML_CERT_PATH", str(BASE_DIR.parent / "uchealth-clinical-sso" / "uchealth-clinical.cer"))
SESSION_SECRET = os.getenv("SAML_SESSION_SECRET", "supplycopia_uchealth_clinical_jwt_secret_key_2026")
COOKIE_NAME = "sc_session"

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')

def _b64url_decode(s: str) -> bytes:
    padding = '=' * (4 - (len(s) % 4))
    return base64.urlsafe_b64decode(s + padding)

class SAMLAuthManager:
    """
    Enterprise SAML 2.0 Single Sign-On & Session Manager for Microsoft Entra ID.
    Replicates SupplyCopia Enrichment Studio SSO architecture.
    """
    def __init__(self):
        self.cert_pem = self._load_cert()

    def _load_cert(self) -> Optional[str]:
        p = Path(CERT_PATH)
        if not p.exists():
            # Try alternate relative path
            alt = BASE_DIR / "uchealth-clinical-sso" / "uchealth-clinical.cer"
            if alt.exists():
                p = alt
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
        return None

    def create_session_token(self, user_info: Dict[str, Any], expires_in: int = 86400 * 7) -> str:
        """
        Creates a tamper-proof signed JWT-style session token.
        """
        now = int(time.time())
        payload = {
            "sub": user_info.get("email") or user_info.get("username") or "uchealth_admin",
            "email": user_info.get("email", ""),
            "name": user_info.get("display_name") or user_info.get("name", "UC Health User"),
            "display_name": user_info.get("display_name") or user_info.get("name", "UC Health User"),
            "given_name": user_info.get("given_name", ""),
            "surname": user_info.get("surname", ""),
            "upn": user_info.get("upn", user_info.get("email", "")),
            "username": user_info.get("username", "uchealth_admin"),
            "role": user_info.get("role", "uchealth_admin"),
            "facility": user_info.get("facility", "UC Health Cincinnati Medical Center"),
            "sso_provider": user_info.get("sso_provider", "Microsoft Entra ID"),
            "iat": now,
            "exp": now + expires_in
        }
        header = {"alg": "HS256", "typ": "JWT"}
        h_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
        p_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        
        token_body = f"{_b64url_encode(h_bytes)}.{_b64url_encode(p_bytes)}"
        sig = hmac.new(SESSION_SECRET.encode('utf-8'), token_body.encode('utf-8'), hashlib.sha256).digest()
        return f"{token_body}.{_b64url_encode(sig)}"

    def verify_session_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifies session token integrity, HMAC signature, and TTL expiration.
        """
        if not token or "." not in token:
            return None
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h_b64, p_b64, sig_b64 = parts
        token_body = f"{h_b64}.{p_b64}"
        expected_sig = hmac.new(SESSION_SECRET.encode('utf-8'), token_body.encode('utf-8'), hashlib.sha256).digest()
        
        try:
            provided_sig = _b64url_decode(sig_b64)
            if not hmac.compare_digest(provided_sig, expected_sig):
                return None
            payload = json.loads(_b64url_decode(p_b64).decode('utf-8'))
            if payload.get("exp", 0) < int(time.time()):
                return None
            return payload
        except Exception:
            return None

    def build_authn_request(self, relay_state: str = "/", acs_override: Optional[str] = None) -> str:
        """
        Generates standard SAML 2.0 AuthnRequest and deflated redirect URL to Microsoft Entra ID.
        """
        authn_id = f"id_{uuid.uuid4().hex}"
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        acs_url = acs_override or SP_ACS_URL

        authn_xml = (
            f'<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol" '
            f'xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion" '
            f'ID="{authn_id}" Version="2.0" IssueInstant="{now_iso}" '
            f'Destination="{IDP_SSO_URL}" AssertionConsumerServiceURL="{acs_url}" '
            f'ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">\n'
            f'  <saml:Issuer>{SP_ENTITY_ID}</saml:Issuer>\n'
            f'  <samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress" AllowCreate="true"/>\n'
            f'</samlp:AuthnRequest>'
        )

        # Standard SAML HTTP-Redirect: raw deflate (rfc1951, strip zlib header & checksum)
        compressed = zlib.compress(authn_xml.encode('utf-8'))[2:-4]
        saml_req_b64 = base64.b64encode(compressed).decode('utf-8')
        
        params = {
            "SAMLRequest": saml_req_b64,
            "RelayState": relay_state or "/"
        }
        return f"{IDP_SSO_URL}?{urllib.parse.urlencode(params)}"

    def parse_saml_response(self, saml_response_b64: str) -> Dict[str, Any]:
        """
        Parses base64-encoded SAMLResponse from Microsoft Entra ID and extracts identity claims.
        """
        try:
            xml_bytes = base64.b64decode(saml_response_b64)
            root = ET.fromstring(xml_bytes)
        except Exception as e:
            raise ValueError(f"Invalid SAMLResponse XML encoding: {str(e)}")

        namespaces = {
            "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
            "saml": "urn:oasis:names:tc:SAML:2.0:assertion"
        }

        # Validate StatusCode
        status_code_el = root.find(".//samlp:StatusCode", namespaces)
        if status_code_el is not None:
            val = status_code_el.attrib.get("Value", "")
            if "Success" not in val:
                raise ValueError(f"SAML authentication rejected by Identity Provider: {val}")

        # Extract Subject / NameID
        name_id_el = root.find(".//saml:Subject/saml:NameID", namespaces)
        name_id = name_id_el.text.strip() if name_id_el is not None and name_id_el.text else ""

        # Extract Claims from AttributeStatement
        claims = {}
        for attr in root.findall(".//saml:AttributeStatement/saml:Attribute", namespaces):
            name = attr.attrib.get("Name", "")
            val_el = attr.find("saml:AttributeValue", namespaces)
            val = val_el.text.strip() if val_el is not None and val_el.text else ""
            if name and val:
                claims[name] = val

        # Map claims standard to Entra ID
        email = (
            claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress") or
            claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name") or
            name_id or
            "user@uchealth.com"
        )
        given_name = claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname", "")
        surname = claims.get("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname", "")
        display_name = (
            claims.get("http://schemas.microsoft.com/identity/claims/displayname") or
            f"{given_name} {surname}".strip() or
            email.split("@")[0].title()
        )
        roles = (
            claims.get("http://schemas.microsoft.com/ws/2008/06/identity/claims/role") or
            "uchealth_admin"
        )

        return {
            "username": email.split("@")[0].lower(),
            "email": email,
            "display_name": display_name,
            "given_name": given_name or "UC Health",
            "surname": surname or "User",
            "upn": email,
            "role": roles,
            "facility": "UC Health Cincinnati Medical Center",
            "sso_provider": "Microsoft Entra ID"
        }

    def get_dev_user(self) -> Dict[str, Any]:
        """
        Returns authorized lead developer profile for localhost bypass testing.
        """
        return {
            "username": "nrayan",
            "email": "nrayan@supplycopia.com",
            "display_name": "Nirmal Rayan",
            "given_name": "Nirmal",
            "surname": "Rayan",
            "upn": "nrayan@supplycopia.com",
            "role": "uchealth_admin",
            "facility": "UC Health Cincinnati Medical Center",
            "sso_provider": "Microsoft Entra ID (Dev Bypass)"
        }

    def generate_sp_metadata(self) -> str:
        """
        Generates standard SP SAML 2.0 Metadata XML.
        """
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata" entityID="{SP_ENTITY_ID}">\n'
            f'  <md:SPSSODescriptor AuthnRequestsSigned="false" WantAssertionsSigned="true" protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">\n'
            f'    <md:SingleLogoutService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" Location="{SP_SLO_URL}"/>\n'
            f'    <md:NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</md:NameIDFormat>\n'
            f'    <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="{SP_ACS_URL}" index="0" isDefault="true"/>\n'
            f'  </md:SPSSODescriptor>\n'
            f'</md:EntityDescriptor>'
        )

_saml_manager_instance: Optional[SAMLAuthManager] = None

def get_saml_manager() -> SAMLAuthManager:
    global _saml_manager_instance
    if _saml_manager_instance is None:
        _saml_manager_instance = SAMLAuthManager()
    return _saml_manager_instance

def get_session_user(request: Request) -> Optional[Dict[str, Any]]:
    """
    Extracts authenticated user from sc_session cookie.
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        # Also check Authorization Bearer header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            
    if token:
        return get_saml_manager().verify_session_token(token)
    return None
