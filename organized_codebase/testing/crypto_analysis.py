"""
Advanced Cryptographic Analysis Module
=====================================

Implements comprehensive cryptographic assessment for Python:
- Cryptographic library usage analysis
- Algorithm strength assessment
- Key management security analysis
- Cryptographic implementation patterns
- SSL/TLS configuration analysis
- Random number generation security
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

from .base_analyzer import BaseAnalyzer


@dataclass
class CryptographicIssue:
    """Represents a cryptographic security issue."""
    issue_id: str
    issue_type: str
    severity: str
    location: Tuple[str, int]
    description: str
    recommendation: str
    confidence: float


@dataclass
class CryptographicAlgorithm:
    """Represents usage of a cryptographic algorithm."""
    algorithm: str
    strength: str  # 'STRONG', 'WEAK', 'DEPRECATED', 'BROKEN'
    usage_context: str
    location: Tuple[str, int]
    library: str


class CryptographicAnalyzer(BaseAnalyzer):
    """Analyzer for cryptographic security assessment."""
    
    def __init__(self, base_path: Path):
        super().__init__(base_path)
        self._init_crypto_patterns()
    
    def _init_crypto_patterns(self):
        """Initialize cryptographic analysis patterns."""
        
        # Cryptographic libraries and their assessment
        self.crypto_libraries = {
            'cryptography': {'strength': 'STRONG', 'recommendation': 'Preferred modern library'},
            'pycryptodome': {'strength': 'STRONG', 'recommendation': 'Good alternative to pycrypto'},
            'pycrypto': {'strength': 'DEPRECATED', 'recommendation': 'Replace with pycryptodome or cryptography'},
            'hashlib': {'strength': 'STRONG', 'recommendation': 'Python standard library'},
            'hmac': {'strength': 'STRONG', 'recommendation': 'Python standard library'},
            'secrets': {'strength': 'STRONG', 'recommendation': 'Use for secure random generation'},
            'ssl': {'strength': 'MEDIUM', 'recommendation': 'Ensure proper configuration'},
            'bcrypt': {'strength': 'STRONG', 'recommendation': 'Excellent for password hashing'},
            'scrypt': {'strength': 'STRONG', 'recommendation': 'Good for key derivation'},
            'passlib': {'strength': 'STRONG', 'recommendation': 'Comprehensive password library'},
            'pyotp': {'strength': 'STRONG', 'recommendation': 'Good for OTP implementation'},
            'jose': {'strength': 'MEDIUM', 'recommendation': 'Check implementation details'},
            'pyjwt': {'strength': 'MEDIUM', 'recommendation': 'Ensure proper key management'}
        }
        
        # Algorithm strength classifications
        self.algorithm_strength = {
            # Strong algorithms
            'AES-256': 'STRONG',
            'AES-192': 'STRONG', 
            'AES-128': 'STRONG',
            'ChaCha20': 'STRONG',
            'RSA-4096': 'STRONG',
            'RSA-2048': 'STRONG',
            'ECDSA-P384': 'STRONG',
            'ECDSA-P256': 'STRONG',
            'SHA-256': 'STRONG',
            'SHA-384': 'STRONG',
            'SHA-512': 'STRONG',
            'SHA-3': 'STRONG',
            'PBKDF2': 'STRONG',
            'scrypt': 'STRONG',
            'bcrypt': 'STRONG',
            'Argon2': 'STRONG',
            
            # Weak but acceptable
            'RSA-1024': 'WEAK',
            'SHA-224': 'WEAK',
            
            # Deprecated algorithms
            'DES': 'DEPRECATED',
            '3DES': 'DEPRECATED',
            'RC4': 'DEPRECATED',
            'MD5': 'DEPRECATED',
            'SHA-1': 'DEPRECATED',
            
            # Broken algorithms
            'MD2': 'BROKEN',
            'MD4': 'BROKEN',
            'RC2': 'BROKEN',
            'DES': 'BROKEN'
        }
        
        # Cryptographic patterns by category
        self.crypto_patterns = {
            'symmetric_encryption': [
                r'AES\.(new|MODE_\w+)',
                r'DES\.(new|MODE_\w+)', 
                r'TripleDES\.(new|MODE_\w+)',
                r'ChaCha20\.',
                r'Salsa20\.',
                r'Blowfish\.',
                r'RC4\.',
                r'Cipher\s*\(',
                r'encrypt\s*\(',
                r'decrypt\s*\(',
            ],
            'asymmetric_encryption': [
                r'RSA\.(generate|import_key|new)',
                r'DSA\.(generate|import_key)',
                r'ECC\.(generate|import_key)',
                r'PKCS1_OAEP\.',
                r'PKCS1_v1_5\.',
                r'public_key\s*\(',
                r'private_key\s*\(',
            ],
            'hashing': [
                r'hashlib\.(md5|sha1|sha224|sha256|sha384|sha512|sha3_\d+)\s*\(',
                r'SHA\d+\.(new|digest)',
                r'MD5\.(new|digest)',
                r'BLAKE2\w*\.',
                r'hash\s*\(',
                r'digest\s*\(',
            ],
            'key_derivation': [
                r'PBKDF2\s*\(',
                r'scrypt\s*\(',
                r'bcrypt\.(gensalt|hashpw|checkpw)',
                r'Argon2\w*\.',
                r'derive\s*\(',
                r'kdf\s*\(',
            ],
            'random_generation': [
                r'random\.(random|randint|choice|shuffle)',
                r'secrets\.(randbits|randbelow|choice|token_\w+)',
                r'os\.urandom\s*\(',
                r'Random\.(random|getrandbits)',
                r'SystemRandom\s*\(',
            ],
            'ssl_tls': [
                r'ssl\.(create_context|wrap_socket|SSLContext)',
                r'ssl\.PROTOCOL_\w+',
                r'ssl\.OP_\w+',
                r'verify_mode\s*=',
                r'check_hostname\s*=',
                r'SSLContext\s*\(',
            ],
            'certificates': [
                r'x509\.',
                r'Certificate\.',
                r'load_pem_\w+_certificate',
                r'load_der_\w+_certificate',
                r'verify\s*\(',
                r'sign\s*\(',
            ],
            'jwt_tokens': [
                r'jwt\.(encode|decode)',
                r'jose\.',
                r'JWK\s*\(',
                r'JWS\s*\(',
                r'JWE\s*\(',
            ]
        }
        
        # Insecure practices patterns
        self.insecure_patterns = {
            'hardcoded_keys': [
                r'key\s*=\s*["\'][^"\']{16,}["\']',
                r'password\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']{8,}["\']',
                r'token\s*=\s*["\'][^"\']{20,}["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
            ],
            'weak_randomness': [
                r'random\.random\s*\(\)',
                r'random\.randint\s*\(',
                r'time\.time\s*\(\)\s*%',
                r'datetime\.now\s*\(\)\.microsecond',
            ],
            'insecure_modes': [
                r'MODE_ECB',
                r'ECB\s*=',
                r'Padding\s*=\s*None',
                r'IV\s*=\s*None',
                r'nonce\s*=\s*None',
            ],
            'ssl_issues': [
                r'verify_mode\s*=\s*ssl\.CERT_NONE',
                r'check_hostname\s*=\s*False',
                r'ssl\._create_unverified_context',
                r'PROTOCOL_SSLv2',
                r'PROTOCOL_SSLv3',
                r'PROTOCOL_TLSv1\s*[^.2]',
            ],
            'weak_algorithms': [
                r'\bMD5\b',
                r'\bSHA1\b', 
                r'\bDES\b',
                r'\bRC4\b',
                r'algorithm\s*=\s*["\']md5["\']',
                r'algorithm\s*=\s*["\']sha1["\']',
            ]
        }
        
        # Key management issues
        self.key_management_patterns = {
            'key_storage': [
                r'key.*\.txt',
                r'private.*\.key\s*=',
                r'store.*key.*file',
                r'save.*key.*disk',
            ],
            'key_transmission': [
                r'send.*key.*http',
                r'post.*key',
                r'get.*key.*url',
                r'transmit.*key',
            ],
            'key_derivation_issues': [
                r'key\s*=\s*password',
                r'key\s*=\s*hash\s*\(',
                r'iterations\s*=\s*[1-9]\d{0,2}',  # Low iteration count
                r'salt\s*=\s*["\'][^"\']{1,8}["\']',  # Short salt
            ]
        }
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive cryptographic analysis."""
        print("[INFO] Performing Cryptographic Security Analysis...")
        
        results = {
            "library_usage": self._analyze_crypto_libraries(),
            "algorithm_assessment": self._assess_algorithms(),
            "cryptographic_issues": self._identify_crypto_issues(),
            "key_management": self._analyze_key_management(),
            "ssl_tls_analysis": self._analyze_ssl_tls(),
            "random_generation": self._analyze_random_generation(),
            "certificate_analysis": self._analyze_certificates(),
            "crypto_metrics": self._calculate_crypto_metrics()
        }
        
        print(f"  [OK] Identified {len(results['cryptographic_issues'])} crypto issues")
        return results
    
    def _analyze_crypto_libraries(self) -> Dict[str, Any]:
        """Analyze usage of cryptographic libraries."""
        library_usage = defaultdict(list)
        import_analysis = defaultdict(int)
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    # Check for imports of crypto libraries
                    if line.strip().startswith(('import ', 'from ')):
                        for lib_name, lib_info in self.crypto_libraries.items():
                            if lib_name in line:
                                library_usage[lib_name].append({
                                    'file': file_key,
                                    'line': line_num,
                                    'import_statement': line.strip(),
                                    'strength': lib_info['strength'],
                                    'recommendation': lib_info['recommendation']
                                })
                                import_analysis[lib_name] += 1
                                
            except Exception:
                continue
        
        return {
            'libraries_found': dict(library_usage),
            'usage_summary': dict(import_analysis),
            'library_assessment': self._assess_library_security(import_analysis)
        }
    
    def _assess_algorithms(self) -> Dict[str, Any]:
        """Assess cryptographic algorithms used in the codebase."""
        algorithms_found = []
        algorithm_counts = defaultdict(int)
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    for category, patterns in self.crypto_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Try to identify specific algorithm
                                algorithm = self._extract_algorithm_from_line(line)
                                strength = self._get_algorithm_strength(algorithm)
                                
                                algorithms_found.append({
                                    'algorithm': algorithm,
                                    'strength': strength,
                                    'category': category,
                                    'location': (file_key, line_num),
                                    'code': line.strip(),
                                    'pattern': pattern
                                })
                                
                                algorithm_counts[algorithm] += 1
                                
            except Exception:
                continue
        
        return {
            'algorithms_detected': algorithms_found,
            'algorithm_distribution': dict(algorithm_counts),
            'strength_summary': self._summarize_algorithm_strength(algorithms_found)
        }
    
    def _identify_crypto_issues(self) -> List[Dict[str, Any]]:
        """Identify cryptographic security issues."""
        issues = []
        issue_id = 1
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    # Check for insecure patterns
                    for issue_type, patterns in self.insecure_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                issues.append({
                                    'issue_id': issue_id,
                                    'type': issue_type,
                                    'severity': self._get_crypto_issue_severity(issue_type),
                                    'location': (file_key, line_num),
                                    'code': line.strip(),
                                    'description': self._get_crypto_issue_description(issue_type),
                                    'recommendation': self._get_crypto_issue_recommendation(issue_type),
                                    'confidence': self._calculate_issue_confidence(pattern, line)
                                })
                                issue_id += 1
                                
            except Exception:
                continue
        
        return issues
    
    def _analyze_key_management(self) -> Dict[str, Any]:
        """Analyze key management practices."""
        key_issues = []
        key_practices = defaultdict(list)
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    for category, patterns in self.key_management_patterns.items():
                        for pattern in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                issue = {
                                    'category': category,
                                    'location': (file_key, line_num),
                                    'code': line.strip(),
                                    'severity': self._get_key_issue_severity(category),
                                    'description': self._get_key_issue_description(category)
                                }
                                
                                key_issues.append(issue)
                                key_practices[category].append(issue)
                                
            except Exception:
                continue
        
        return {
            'key_management_issues': key_issues,
            'issue_categories': dict(key_practices),
            'key_security_score': self._calculate_key_security_score(key_issues)
        }
    
    def _analyze_ssl_tls(self) -> Dict[str, Any]:
        """Analyze SSL/TLS configuration and usage."""
        ssl_usage = []
        ssl_issues = []
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    # Look for SSL/TLS usage
                    for pattern in self.crypto_patterns['ssl_tls']:
                        if re.search(pattern, line, re.IGNORECASE):
                            ssl_config = self._analyze_ssl_config(line)
                            
                            ssl_usage.append({
                                'location': (file_key, line_num),
                                'code': line.strip(),
                                'configuration': ssl_config,
                                'security_level': ssl_config.get('security_level', 'UNKNOWN')
                            })
                            
                            # Check for SSL issues
                            if ssl_config.get('security_level') == 'INSECURE':
                                ssl_issues.append({
                                    'location': (file_key, line_num),
                                    'issue': ssl_config.get('issue', 'SSL security issue'),
                                    'severity': 'HIGH',
                                    'recommendation': ssl_config.get('recommendation', 'Review SSL configuration')
                                })
                                
            except Exception:
                continue
        
        return {
            'ssl_usage': ssl_usage,
            'ssl_issues': ssl_issues,
            'ssl_security_summary': self._summarize_ssl_security(ssl_usage, ssl_issues)
        }
    
    def _analyze_random_generation(self) -> Dict[str, Any]:
        """Analyze random number generation security."""
        random_usage = []
        weak_randomness = []
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    # Check secure random generation
                    for pattern in self.crypto_patterns['random_generation']:
                        if re.search(pattern, line, re.IGNORECASE):
                            security_level = self._assess_random_security(line)
                            
                            random_usage.append({
                                'location': (file_key, line_num),
                                'code': line.strip(),
                                'security_level': security_level,
                                'function': self._extract_random_function(line)
                            })
                            
                            if security_level == 'WEAK':
                                weak_randomness.append({
                                    'location': (file_key, line_num),
                                    'code': line.strip(),
                                    'issue': 'Weak random number generation',
                                    'recommendation': 'Use secrets module for cryptographic randomness'
                                })
                                
            except Exception:
                continue
        
        return {
            'random_usage': random_usage,
            'weak_randomness_issues': weak_randomness,
            'randomness_security_score': self._calculate_randomness_score(random_usage)
        }
    
    def _analyze_certificates(self) -> Dict[str, Any]:
        """Analyze certificate handling and validation."""
        cert_usage = []
        cert_issues = []
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                file_key = str(py_file.relative_to(self.base_path))
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.crypto_patterns['certificates']:
                        if re.search(pattern, line, re.IGNORECASE):
                            cert_analysis = self._analyze_certificate_usage(line)
                            
                            cert_usage.append({
                                'location': (file_key, line_num),
                                'code': line.strip(),
                                'usage_type': cert_analysis.get('type', 'UNKNOWN'),
                                'security_assessment': cert_analysis.get('security', 'UNKNOWN')
                            })
                            
                            if cert_analysis.get('security') == 'INSECURE':
                                cert_issues.append({
                                    'location': (file_key, line_num),
                                    'issue': cert_analysis.get('issue', 'Certificate security issue'),
                                    'severity': 'HIGH',
                                    'recommendation': cert_analysis.get('recommendation', 'Review certificate handling')
                                })
                                
            except Exception:
                continue
        
        return {
            'certificate_usage': cert_usage,
            'certificate_issues': cert_issues,
            'certificate_security_summary': len(cert_issues) == 0
        }
    
    def _calculate_crypto_metrics(self) -> Dict[str, Any]:
        """Calculate overall cryptographic security metrics."""
        # Get all analysis results for metrics calculation
        issues = self._identify_crypto_issues()
        key_mgmt = self._analyze_key_management()
        ssl_analysis = self._analyze_ssl_tls()
        random_analysis = self._analyze_random_generation()
        
        # Count issues by severity
        issue_counts = Counter()
        for issue in issues:
            issue_counts[issue['severity']] += 1
        
        # Calculate overall crypto score (0-100)
        total_issues = len(issues)
        key_issues = len(key_mgmt.get('key_management_issues', []))
        ssl_issues = len(ssl_analysis.get('ssl_issues', []))
        random_issues = len(random_analysis.get('weak_randomness_issues', []))
        
        crypto_score = max(0, 100 - 
                          (issue_counts['CRITICAL'] * 25) -
                          (issue_counts['HIGH'] * 15) -
                          (issue_counts['MEDIUM'] * 8) -
                          (issue_counts['LOW'] * 3) -
                          (key_issues * 10) -
                          (ssl_issues * 12) -
                          (random_issues * 8))
        
        return {
            'total_crypto_issues': total_issues,
            'critical_issues': issue_counts['CRITICAL'],
            'high_issues': issue_counts['HIGH'], 
            'medium_issues': issue_counts['MEDIUM'],
            'low_issues': issue_counts['LOW'],
            'key_management_issues': key_issues,
            'ssl_issues': ssl_issues,
            'randomness_issues': random_issues,
            'crypto_security_score': crypto_score,
            'crypto_grade': self._get_crypto_grade(crypto_score),
            'improvement_priority': self._get_improvement_priorities(issues, key_mgmt, ssl_analysis, random_analysis)
        }
    
    # Helper methods for detailed analysis
    
    def _extract_algorithm_from_line(self, line: str) -> str:
        """Extract algorithm name from code line."""
        # Simple pattern matching for common algorithms
        algorithms = ['AES', 'DES', 'RSA', 'SHA256', 'SHA1', 'MD5', 'PBKDF2', 'bcrypt', 'scrypt']
        for alg in algorithms:
            if alg.lower() in line.lower():
                return alg
        return 'UNKNOWN'
    
    def _get_algorithm_strength(self, algorithm: str) -> str:
        """Get strength assessment for algorithm."""
        return self.algorithm_strength.get(algorithm, 'UNKNOWN')
    
    def _get_crypto_issue_severity(self, issue_type: str) -> str:
        """Get severity for crypto issue type."""
        severity_map = {
            'hardcoded_keys': 'CRITICAL',
            'weak_algorithms': 'HIGH',
            'insecure_modes': 'HIGH',
            'ssl_issues': 'HIGH',
            'weak_randomness': 'MEDIUM'
        }
        return severity_map.get(issue_type, 'MEDIUM')
    
    def _get_crypto_issue_description(self, issue_type: str) -> str:
        """Get description for crypto issue type."""
        descriptions = {
            'hardcoded_keys': 'Hardcoded cryptographic keys or secrets detected',
            'weak_algorithms': 'Use of weak or deprecated cryptographic algorithms',
            'insecure_modes': 'Use of insecure cryptographic modes or configurations',
            'ssl_issues': 'SSL/TLS security configuration issues',
            'weak_randomness': 'Use of weak random number generation for security purposes'
        }
        return descriptions.get(issue_type, 'Cryptographic security issue detected')
    
    def _get_crypto_issue_recommendation(self, issue_type: str) -> str:
        """Get recommendation for crypto issue type."""
        recommendations = {
            'hardcoded_keys': 'Use environment variables or secure key management systems',
            'weak_algorithms': 'Replace with strong algorithms: AES-256, RSA-2048+, SHA-256+',
            'insecure_modes': 'Use secure modes: CBC with IV, GCM, or modern AEAD ciphers',
            'ssl_issues': 'Use TLS 1.2+, enable certificate verification, disable weak ciphers',
            'weak_randomness': 'Use secrets module or os.urandom() for cryptographic purposes'
        }
        return recommendations.get(issue_type, 'Review and improve cryptographic implementation')
    
    def _calculate_issue_confidence(self, pattern: str, line: str) -> float:
        """Calculate confidence level for detected issue."""
        # Higher confidence for more specific patterns
        if 'key.*=' in pattern and ('password' in line or 'secret' in line):
            return 0.9
        elif 'MD5' in pattern or 'SHA1' in pattern:
            return 0.8
        else:
            return 0.7
    
    def _assess_library_security(self, usage: Dict[str, int]) -> Dict[str, Any]:
        """Assess overall security of crypto libraries used."""
        strong_libs = sum(count for lib, count in usage.items() 
                         if self.crypto_libraries.get(lib, {}).get('strength') == 'STRONG')
        weak_libs = sum(count for lib, count in usage.items() 
                       if self.crypto_libraries.get(lib, {}).get('strength') in ['WEAK', 'DEPRECATED'])
        
        total_usage = sum(usage.values())
        strength_ratio = strong_libs / max(total_usage, 1)
        
        return {
            'strong_library_usage': strong_libs,
            'weak_library_usage': weak_libs,
            'total_usage': total_usage,
            'strength_ratio': strength_ratio,
            'security_level': 'GOOD' if strength_ratio > 0.8 else 'MEDIUM' if strength_ratio > 0.5 else 'POOR'
        }
    
    def _summarize_algorithm_strength(self, algorithms: List[Dict]) -> Dict[str, Any]:
        """Summarize strength distribution of algorithms."""
        strength_counts = Counter()
        for alg in algorithms:
            strength_counts[alg['strength']] += 1
        
        total = sum(strength_counts.values())
        return {
            'strong_algorithms': strength_counts['STRONG'],
            'weak_algorithms': strength_counts['WEAK'], 
            'deprecated_algorithms': strength_counts['DEPRECATED'],
            'broken_algorithms': strength_counts['BROKEN'],
            'total_algorithms': total,
            'strength_ratio': strength_counts['STRONG'] / max(total, 1)
        }
    
    def _get_key_issue_severity(self, category: str) -> str:
        """Get severity for key management issue."""
        severity_map = {
            'key_storage': 'HIGH',
            'key_transmission': 'CRITICAL',
            'key_derivation_issues': 'MEDIUM'
        }
        return severity_map.get(category, 'MEDIUM')
    
    def _get_key_issue_description(self, category: str) -> str:
        """Get description for key management issue."""
        descriptions = {
            'key_storage': 'Insecure key storage detected',
            'key_transmission': 'Insecure key transmission detected',
            'key_derivation_issues': 'Weak key derivation parameters detected'
        }
        return descriptions.get(category, 'Key management security issue')
    
    def _calculate_key_security_score(self, issues: List[Dict]) -> float:
        """Calculate key management security score."""
        if not issues:
            return 100.0
        
        score = 100.0
        for issue in issues:
            if issue['severity'] == 'CRITICAL':
                score -= 30
            elif issue['severity'] == 'HIGH':
                score -= 20
            elif issue['severity'] == 'MEDIUM':
                score -= 10
        
        return max(0.0, score)
    
    def _analyze_ssl_config(self, line: str) -> Dict[str, Any]:
        """Analyze SSL configuration from code line."""
        config = {'security_level': 'UNKNOWN'}
        
        if 'CERT_NONE' in line or 'verify_mode.*False' in line:
            config = {
                'security_level': 'INSECURE',
                'issue': 'SSL certificate verification disabled',
                'recommendation': 'Enable certificate verification'
            }
        elif 'PROTOCOL_SSLv' in line or 'PROTOCOL_TLSv1' in line:
            if 'TLSv1_2' not in line and 'TLSv1_3' not in line:
                config = {
                    'security_level': 'INSECURE',
                    'issue': 'Weak SSL/TLS protocol version',
                    'recommendation': 'Use TLS 1.2 or higher'
                }
        else:
            config['security_level'] = 'SECURE'
        
        return config
    
    def _summarize_ssl_security(self, usage: List[Dict], issues: List[Dict]) -> Dict[str, Any]:
        """Summarize SSL security status."""
        total_ssl_usage = len(usage)
        ssl_issues_count = len(issues)
        
        return {
            'total_ssl_usage': total_ssl_usage,
            'ssl_issues_found': ssl_issues_count,
            'ssl_security_ratio': (total_ssl_usage - ssl_issues_count) / max(total_ssl_usage, 1),
            'ssl_grade': 'GOOD' if ssl_issues_count == 0 else 'POOR'
        }
    
    def _assess_random_security(self, line: str) -> str:
        """Assess security of random number generation."""
        if 'secrets.' in line or 'os.urandom' in line or 'SystemRandom' in line:
            return 'STRONG'
        elif 'random.random' in line or 'random.randint' in line:
            return 'WEAK'
        else:
            return 'MEDIUM'
    
    def _extract_random_function(self, line: str) -> str:
        """Extract random function name from line."""
        if 'secrets.' in line:
            return 'secrets'
        elif 'os.urandom' in line:
            return 'os.urandom'
        elif 'random.' in line:
            return 'random'
        else:
            return 'unknown'
    
    def _calculate_randomness_score(self, usage: List[Dict]) -> float:
        """Calculate randomness security score."""
        if not usage:
            return 100.0
        
        strong_count = sum(1 for u in usage if u['security_level'] == 'STRONG')
        total_count = len(usage)
        
        return (strong_count / total_count) * 100.0
    
    def _analyze_certificate_usage(self, line: str) -> Dict[str, Any]:
        """Analyze certificate usage from code line."""
        return {
            'type': 'certificate_operation',
            'security': 'SECURE',  # Default assumption
        }
    
    def _get_crypto_grade(self, score: float) -> str:
        """Get crypto security grade based on score."""
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _get_improvement_priorities(self, issues: List[Dict], key_mgmt: Dict, 
                                  ssl_analysis: Dict, random_analysis: Dict) -> List[str]:
        """Get prioritized list of improvements needed."""
        priorities = []
        
        # High priority issues
        critical_issues = [i for i in issues if i['severity'] == 'CRITICAL']
        if critical_issues:
            priorities.append('Fix critical cryptographic vulnerabilities')
        
        high_issues = [i for i in issues if i['severity'] == 'HIGH']
        if high_issues:
            priorities.append('Address high-severity crypto issues')
        
        if key_mgmt.get('key_management_issues'):
            priorities.append('Improve key management practices')
        
        if ssl_analysis.get('ssl_issues'):
            priorities.append('Fix SSL/TLS configuration issues')
        
        if random_analysis.get('weak_randomness_issues'):
            priorities.append('Replace weak random number generation')
        
        # Medium priority
        medium_issues = [i for i in issues if i['severity'] == 'MEDIUM']
        if medium_issues:
            priorities.append('Resolve medium-severity crypto issues')
        
        return priorities[:5]  # Return top 5 priorities