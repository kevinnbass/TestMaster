"""
Agency-Swarm Derived File Security Module
Extracted from agency-swarm/agency_swarm/util/files.py
Enhanced for comprehensive file security handling
"""

import os
import mimetypes
import hashlib
import magic
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from .error_handler import ValidationError, SecurityError, security_error_handler


# Enhanced MIME type definitions based on agency-swarm patterns
SAFE_IMAGE_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif", "image/bmp"
}

SAFE_DOCUMENT_TYPES = {
    "application/pdf", "text/plain", "text/markdown", "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
}

SAFE_CODE_TYPES = {
    "text/x-python", "text/x-java", "text/javascript", "text/x-c", "text/x-csharp",
    "text/x-c++", "text/html", "text/css", "application/json", "text/xml"
}

DANGEROUS_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
    '.ps1', '.sh', '.bin', '.msi', '.deb', '.rpm'
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB default


@dataclass
class FileSecurityResult:
    """File security validation result"""
    is_safe: bool
    file_type: str
    mime_type: str
    size: int
    hash_sha256: str
    risk_level: str
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class FileSecurityValidator:
    """Comprehensive file security validation system"""
    
    def __init__(self, max_file_size: int = MAX_FILE_SIZE):
        self.max_file_size = max_file_size
        self.logger = logging.getLogger(__name__)
        self.quarantine_dir = Path("quarantine")
        self.quarantine_dir.mkdir(exist_ok=True)
        
        # Initialize MIME types (from agency-swarm pattern)
        mimetypes.add_type(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"
        )
        mimetypes.add_type(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"
        )
        mimetypes.add_type(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"
        )
        
    def validate_file(self, file_path: str) -> FileSecurityResult:
        """Comprehensive file security validation"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                return FileSecurityResult(
                    is_safe=False,
                    file_type="unknown",
                    mime_type="unknown",
                    size=0,
                    hash_sha256="",
                    risk_level="high",
                    issues=["File does not exist"],
                    recommendations=["Verify file path"]
                )
            
            # Get file stats
            file_size = path.stat().st_size
            file_hash = self._calculate_hash(file_path)
            
            # Basic size check
            if file_size > self.max_file_size:
                return FileSecurityResult(
                    is_safe=False,
                    file_type=path.suffix,
                    mime_type="unknown",
                    size=file_size,
                    hash_sha256=file_hash,
                    risk_level="high",
                    issues=[f"File too large: {file_size} bytes"],
                    recommendations=["Reduce file size or contact administrator"]
                )
            
            # MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                try:
                    mime_type = magic.from_file(file_path, mime=True)
                except:
                    mime_type = "application/octet-stream"
            
            # Extension validation
            extension = path.suffix.lower()
            issues = []
            recommendations = []
            risk_level = "low"
            
            # Check for dangerous extensions
            if extension in DANGEROUS_EXTENSIONS:
                issues.append(f"Dangerous file extension: {extension}")
                recommendations.append("File type not permitted for upload")
                risk_level = "critical"
            
            # MIME type validation
            is_safe_type = (
                mime_type in SAFE_IMAGE_TYPES or 
                mime_type in SAFE_DOCUMENT_TYPES or 
                mime_type in SAFE_CODE_TYPES
            )
            
            if not is_safe_type and extension not in DANGEROUS_EXTENSIONS:
                issues.append(f"Unrecognized or potentially unsafe MIME type: {mime_type}")
                recommendations.append("File type requires manual review")
                risk_level = "medium"
            
            # Content validation
            content_issues = self._validate_file_content(file_path, mime_type)
            issues.extend(content_issues)
            
            if content_issues:
                risk_level = "high"
                recommendations.append("File content requires sanitization")
            
            is_safe = len(issues) == 0 and is_safe_type
            
            result = FileSecurityResult(
                is_safe=is_safe,
                file_type=extension,
                mime_type=mime_type,
                size=file_size,
                hash_sha256=file_hash,
                risk_level=risk_level,
                issues=issues,
                recommendations=recommendations
            )
            
            self.logger.info(f"File validation complete: {file_path} - Safe: {is_safe}")
            return result
            
        except Exception as e:
            error = SecurityError(f"File validation failed: {str(e)}")
            security_error_handler.handle_error(error)
            
            return FileSecurityResult(
                is_safe=False,
                file_type="unknown",
                mime_type="unknown",
                size=0,
                hash_sha256="",
                risk_level="critical",
                issues=[f"Validation error: {str(e)}"],
                recommendations=["Contact system administrator"]
            )
    
    def get_file_purpose(self, file_path: str) -> str:
        """Determine file purpose based on agency-swarm pattern"""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                raise ValidationError(f"Could not determine type for file: {file_path}")
                
            if mime_type in SAFE_IMAGE_TYPES:
                return "vision"
            if (mime_type in SAFE_DOCUMENT_TYPES or 
                mime_type in SAFE_CODE_TYPES):
                return "assistants"
                
            raise ValidationError(f"Unsupported file type: {mime_type}")
            
        except Exception as e:
            error = ValidationError(f"File purpose detection failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def get_safe_tools(self, file_path: str) -> List[Dict[str, str]]:
        """Get safe tools for file processing based on agency-swarm pattern"""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                raise ValidationError(f"Could not determine type for file: {file_path}")
            
            # Validate file first
            validation_result = self.validate_file(file_path)
            if not validation_result.is_safe:
                raise SecurityError(f"File failed security validation: {validation_result.issues}")
            
            safe_code_interpreter_types = SAFE_DOCUMENT_TYPES | SAFE_CODE_TYPES
            
            if mime_type in safe_code_interpreter_types:
                return [{"type": "code_interpreter"}]
            elif mime_type in SAFE_DOCUMENT_TYPES:
                return [{"type": "code_interpreter"}, {"type": "file_search"}]
            else:
                raise ValidationError(f"No safe tools available for file type: {mime_type}")
                
        except Exception as e:
            error = SecurityError(f"Tool determination failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def quarantine_file(self, file_path: str, reason: str) -> str:
        """Move potentially dangerous file to quarantine"""
        try:
            source_path = Path(file_path)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"{timestamp}_{source_path.name}"
            quarantine_path = self.quarantine_dir / quarantine_name
            
            # Move file to quarantine
            source_path.rename(quarantine_path)
            
            # Log quarantine action
            self.logger.warning(f"File quarantined: {file_path} -> {quarantine_path} (Reason: {reason})")
            
            return str(quarantine_path)
            
        except Exception as e:
            error = SecurityError(f"File quarantine failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return "hash_calculation_failed"
    
    def _validate_file_content(self, file_path: str, mime_type: str) -> List[str]:
        """Validate file content for security issues"""
        issues = []
        
        try:
            # For text-based files, scan for malicious content
            if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # Read first 10KB
                    
                    # Check for script injection attempts
                    dangerous_patterns = [
                        r'<script[^>]*>',
                        r'javascript:',
                        r'data:text/html',
                        r'eval\s*\(',
                        r'exec\s*\(',
                        r'system\s*\(',
                        r'shell_exec\s*\('
                    ]
                    
                    import re
                    for pattern in dangerous_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues.append(f"Detected potentially malicious pattern: {pattern}")
            
        except Exception as e:
            issues.append(f"Content validation error: {str(e)}")
        
        return issues


class SecureFileHandler:
    """Secure file operations wrapper"""
    
    def __init__(self):
        self.validator = FileSecurityValidator()
        self.logger = logging.getLogger(__name__)
    
    def secure_upload(self, file_path: str, destination_dir: str) -> Tuple[bool, str]:
        """Securely handle file upload"""
        try:
            # Validate file
            validation_result = self.validator.validate_file(file_path)
            
            if not validation_result.is_safe:
                # Quarantine dangerous file
                quarantine_path = self.validator.quarantine_file(
                    file_path, 
                    f"Security validation failed: {validation_result.issues}"
                )
                return False, f"File quarantined due to security issues: {validation_result.issues}"
            
            # Safe to process
            dest_path = Path(destination_dir) / Path(file_path).name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to destination
            import shutil
            shutil.copy2(file_path, dest_path)
            
            self.logger.info(f"File securely uploaded: {file_path} -> {dest_path}")
            return True, str(dest_path)
            
        except Exception as e:
            error = SecurityError(f"Secure upload failed: {str(e)}")
            security_error_handler.handle_error(error)
            return False, str(error)
    
    def get_file_info(self, file_path: str) -> Dict[str, any]:
        """Get comprehensive file information"""
        validation_result = self.validator.validate_file(file_path)
        
        return {
            'path': file_path,
            'is_safe': validation_result.is_safe,
            'mime_type': validation_result.mime_type,
            'size': validation_result.size,
            'hash': validation_result.hash_sha256,
            'risk_level': validation_result.risk_level,
            'issues': validation_result.issues,
            'recommendations': validation_result.recommendations,
            'purpose': self.validator.get_file_purpose(file_path) if validation_result.is_safe else None,
            'safe_tools': self.validator.get_safe_tools(file_path) if validation_result.is_safe else []
        }


# Global file handler instance
secure_file_handler = SecureFileHandler()