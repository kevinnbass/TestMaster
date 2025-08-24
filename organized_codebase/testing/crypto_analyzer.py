"""
Cryptographic Security Analyzer

Detects weak cryptographic implementations and insecure practices.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CryptoIssueType(Enum):
    """Types of cryptographic issues."""
    WEAK_ALGORITHM = "weak_algorithm"
    WEAK_KEY_SIZE = "weak_key_size"
    INSECURE_RANDOM = "insecure_random"
    HARDCODED_KEY = "hardcoded_key"
    WEAK_HASH = "weak_hash"
    NO_SALT = "no_salt"
    ECB_MODE = "ecb_mode"
    WEAK_PADDING = "weak_padding"
    

@dataclass
class CryptoIssue:
    """Represents a cryptographic security issue."""
    type: CryptoIssueType
    severity: str  # critical, high, medium, low
    file_path: str
    line_number: int
    algorithm: Optional[str]
    description: str
    recommendation: str
    cwe_id: Optional[str]
    

class CryptoAnalyzer:
    """
    Analyzes code for cryptographic weaknesses and insecure practices.
    Detects weak algorithms, insufficient key sizes, and implementation flaws.
    """
    
    def __init__(self):
        """Initialize the crypto analyzer."""
        self.issues = []
        self.files_analyzed = 0
        self.crypto_imports = set()
        logger.info("Crypto Analyzer initialized")
        
    def analyze_file(self, file_path: str) -> List[CryptoIssue]:
        """
        Analyze a file for cryptographic issues.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of crypto issues found
        """
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            # Detect crypto library imports
            self._detect_crypto_imports(tree)
            
            # Check for weak algorithms
            issues.extend(self._check_weak_algorithms(content, file_path))
            
            # Check for hardcoded keys
            issues.extend(self._check_hardcoded_keys(content, file_path))
            
            # Check for insecure random
            issues.extend(self._check_insecure_random(tree, file_path))
            
            # Check for weak hashing
            issues.extend(self._check_weak_hashing(content, file_path))
            
            # Check for ECB mode
            issues.extend(self._check_ecb_mode(content, file_path))
            
            # Check key sizes
            issues.extend(self._check_key_sizes(content, file_path))
            
            self.files_analyzed += 1
            self.issues.extend(issues)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            
        return issues
        
    def _detect_crypto_imports(self, tree: ast.AST) -> None:
        """Detect cryptographic library imports."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(crypto in alias.name for crypto in 
                          ['crypto', 'hashlib', 'hmac', 'secrets', 'ssl']):
                        self.crypto_imports.add(alias.name)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(crypto in node.module for crypto in
                                      ['cryptography', 'Crypto', 'hashlib', 'ssl']):
                    self.crypto_imports.add(node.module)
                    
    def _check_weak_algorithms(self, content: str, file_path: str) -> List[CryptoIssue]:
        """Check for weak cryptographic algorithms."""
        issues = []
        
        weak_algos = {
            'DES': ('DES is cryptographically broken', 'Use AES instead'),
            '3DES': ('3DES is deprecated', 'Use AES instead'),
            'RC4': ('RC4 is insecure', 'Use AES-GCM or ChaCha20'),
            'MD5': ('MD5 is cryptographically broken', 'Use SHA-256 or SHA-3'),
            'SHA1': ('SHA-1 is deprecated for security', 'Use SHA-256 or SHA-3'),
            'ECB': ('ECB mode is insecure', 'Use CBC, CTR, or GCM mode'),
        }
        
        for algo, (desc, rec) in weak_algos.items():
            pattern = re.compile(rf'\b{algo}\b', re.IGNORECASE)
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                issues.append(CryptoIssue(
                    type=CryptoIssueType.WEAK_ALGORITHM,
                    severity="high" if algo in ['DES', 'RC4', 'MD5'] else "medium",
                    file_path=file_path,
                    line_number=line_num,
                    algorithm=algo,
                    description=desc,
                    recommendation=rec,
                    cwe_id="CWE-327"
                ))
                
        return issues
        
    def _check_hardcoded_keys(self, content: str, file_path: str) -> List[CryptoIssue]:
        """Check for hardcoded cryptographic keys."""
        issues = []
        
        # Patterns for potential hardcoded keys
        patterns = [
            (r'["\'](?:key|secret|password)["\']?\s*=\s*["\'][A-Za-z0-9+/=]{16,}["\']', 'Hardcoded key'),
            (r'AES\.new\(["\'][^"\']+["\']', 'Hardcoded AES key'),
            (r'Fernet\(["\'][^"\']+["\']', 'Hardcoded Fernet key'),
            (r'private_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded private key'),
        ]
        
        for pattern, desc in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                line_num = content[:match.start()].count('\n') + 1
                issues.append(CryptoIssue(
                    type=CryptoIssueType.HARDCODED_KEY,
                    severity="critical",
                    file_path=file_path,
                    line_number=line_num,
                    algorithm=None,
                    description=f"{desc} detected",
                    recommendation="Use environment variables or key management service",
                    cwe_id="CWE-798"
                ))
                
        return issues
        
    def _check_insecure_random(self, tree: ast.AST, file_path: str) -> List[CryptoIssue]:
        """Check for use of insecure random for crypto."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == 'random':
                        # Check if used for crypto
                        issues.append(CryptoIssue(
                            type=CryptoIssueType.INSECURE_RANDOM,
                            severity="high",
                            file_path=file_path,
                            line_number=node.lineno,
                            algorithm="random",
                            description="Using 'random' module which is not cryptographically secure",
                            recommendation="Use 'secrets' module for cryptographic randomness",
                            cwe_id="CWE-338"
                        ))
                        
        return issues
        
    def _check_weak_hashing(self, content: str, file_path: str) -> List[CryptoIssue]:
        """Check for weak hashing practices."""
        issues = []
        
        # Check for hashing without salt
        if 'hashlib' in content and 'salt' not in content.lower():
            issues.append(CryptoIssue(
                type=CryptoIssueType.NO_SALT,
                severity="medium",
                file_path=file_path,
                line_number=0,
                algorithm="hash",
                description="Hashing without salt detected",
                recommendation="Always use salt when hashing passwords",
                cwe_id="CWE-759"
            ))
            
        # Check for weak hash functions
        weak_hashes = ['md5', 'sha1']
        for hash_func in weak_hashes:
            pattern = f'hashlib\\.{hash_func}'
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(CryptoIssue(
                    type=CryptoIssueType.WEAK_HASH,
                    severity="high",
                    file_path=file_path,
                    line_number=0,
                    algorithm=hash_func.upper(),
                    description=f"Weak hash function {hash_func.upper()} used",
                    recommendation="Use SHA-256, SHA-3, or bcrypt for passwords",
                    cwe_id="CWE-328"
                ))
                
        return issues
        
    def _check_ecb_mode(self, content: str, file_path: str) -> List[CryptoIssue]:
        """Check for ECB mode usage."""
        issues = []
        
        ecb_patterns = [
            r'AES\.MODE_ECB',
            r'DES\.MODE_ECB',
            r'mode\s*=\s*["\']ECB["\']',
        ]
        
        for pattern in ecb_patterns:
            if re.search(pattern, content):
                issues.append(CryptoIssue(
                    type=CryptoIssueType.ECB_MODE,
                    severity="high",
                    file_path=file_path,
                    line_number=0,
                    algorithm="ECB",
                    description="ECB mode is insecure for encryption",
                    recommendation="Use CBC, CTR, or GCM mode instead",
                    cwe_id="CWE-327"
                ))
                break
                
        return issues
        
    def _check_key_sizes(self, content: str, file_path: str) -> List[CryptoIssue]:
        """Check for weak key sizes."""
        issues = []
        
        # Check for weak RSA key sizes
        rsa_pattern = r'RSA\.generate\((\d+)\)'
        for match in re.finditer(rsa_pattern, content):
            key_size = int(match.group(1))
            if key_size < 2048:
                line_num = content[:match.start()].count('\n') + 1
                issues.append(CryptoIssue(
                    type=CryptoIssueType.WEAK_KEY_SIZE,
                    severity="high" if key_size < 1024 else "medium",
                    file_path=file_path,
                    line_number=line_num,
                    algorithm="RSA",
                    description=f"Weak RSA key size: {key_size} bits",
                    recommendation="Use at least 2048-bit RSA keys",
                    cwe_id="CWE-326"
                ))
                
        # Check for weak AES key sizes
        aes_pattern = r'get_random_bytes\((\d+)\)'
        for match in re.finditer(aes_pattern, content):
            key_size = int(match.group(1))
            if key_size < 16:  # Less than 128 bits
                line_num = content[:match.start()].count('\n') + 1
                issues.append(CryptoIssue(
                    type=CryptoIssueType.WEAK_KEY_SIZE,
                    severity="high",
                    file_path=file_path,
                    line_number=line_num,
                    algorithm="AES",
                    description=f"Weak key size: {key_size * 8} bits",
                    recommendation="Use at least 128-bit keys (16 bytes)",
                    cwe_id="CWE-326"
                ))
                
        return issues
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate cryptographic analysis report.
        
        Returns:
            Analysis report
        """
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        type_counts = {}
        
        for issue in self.issues:
            severity_counts[issue.severity] += 1
            type_name = issue.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
        return {
            'files_analyzed': self.files_analyzed,
            'total_issues': len(self.issues),
            'crypto_libraries_detected': list(self.crypto_imports),
            'issues_by_severity': severity_counts,
            'issues_by_type': type_counts,
            'critical_issues': [
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.type.value,
                    'description': issue.description,
                    'recommendation': issue.recommendation
                }
                for issue in self.issues if issue.severity == 'critical'
            ]
        }
        
    def get_recommendations(self) -> List[str]:
        """
        Get security recommendations based on analysis.
        
        Returns:
            List of recommendations
        """
        recommendations = set()
        
        for issue in self.issues:
            recommendations.add(issue.recommendation)
            
        # Add general recommendations
        if self.crypto_imports:
            recommendations.add("Regularly update cryptographic libraries")
            recommendations.add("Use established crypto libraries, don't roll your own")
            recommendations.add("Follow OWASP cryptographic storage cheat sheet")
            
        return list(recommendations)