import re
import numpy as np
from urllib.parse import urlparse
from collections import Counter
import pandas as pd

def calculate_entropy(string):
    char_counts = Counter(string)
    total_chars = len(string)
    probabilities = [count / total_chars for count in char_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def extract_features(url):
    features = {}
    
    # Parse the URL
    print(url)
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    query = parsed_url.query
    fragment = parsed_url.fragment
    
    # URL length
    features['url_length'] = len(url)
    
    # Number of dots in URL
    features['number_of_dots_in_url'] = url.count('.')
    
    # Having repeated digits in URL
    features['having_repeated_digits_in_url'] = int(bool(re.search(r'(\d)\1', url)))
    
    # Number of digits in URL
    features['number_of_digits_in_url'] = sum(c.isdigit() for c in url)
    
    # Number of special characters in URL
    special_chars = re.escape('!@#$%^&*()_+-=[]{}|;:\'",.<>?/`~')
    features['number_of_special_char_in_url'] = len(re.findall(f'[{special_chars}]', url))
    
    # Number of hyphens in URL
    features['number_of_hyphens_in_url'] = url.count('-')
    
    # Number of underline in URL
    features['number_of_underline_in_url'] = url.count('_')
    
    # Number of slashes in URL
    features['number_of_slash_in_url'] = url.count('/')
    
    # Number of question marks in URL
    features['number_of_questionmark_in_url'] = url.count('?')
    
    # Number of equal signs in URL
    features['number_of_equal_in_url'] = url.count('=')
    
    # Number of at signs in URL
    features['number_of_at_in_url'] = url.count('@')
    
    # Number of dollar signs in URL
    features['number_of_dollar_in_url'] = url.count('$')
    
    # Number of exclamation marks in URL
    features['number_of_exclamation_in_url'] = url.count('!')
    
    # Number of hashtags in URL
    features['number_of_hashtag_in_url'] = url.count('#')
    
    # Number of percent signs in URL
    features['number_of_percent_in_url'] = url.count('%')
    
    # Domain length
    features['domain_length'] = len(domain)
    
    # Number of dots in domain
    features['number_of_dots_in_domain'] = domain.count('.')
    
    # Number of hyphens in domain
    features['number_of_hyphens_in_domain'] = domain.count('-')
    
    # Having special characters in domain
    features['having_special_characters_in_domain'] = int(bool(re.search(f'[{special_chars}]', domain)))
    
    # Number of special characters in domain
    features['number_of_special_characters_in_domain'] = len(re.findall(f'[{special_chars}]', domain))
    
    # Having digits in domain
    features['having_digits_in_domain'] = int(any(c.isdigit() for c in domain))
    
    # Number of digits in domain
    features['number_of_digits_in_domain'] = sum(c.isdigit() for c in domain)
    
    # Having repeated digits in domain
    features['having_repeated_digits_in_domain'] = int(bool(re.search(r'(\d)\1', domain)))
    
    # Number of subdomains
    features['number_of_subdomains'] = domain.count('.') - 1
    
    # Having dot in subdomain
    features['having_dot_in_subdomain'] = int(any('.' in sub for sub in domain.split('.')[:-2]))
    
    # Having hyphen in subdomain
    features['having_hyphen_in_subdomain'] = int(any('-' in sub for sub in domain.split('.')[:-2]))
    
    # Average subdomain length
    subdomains = domain.split('.')[:-2]
    features['average_subdomain_length'] = np.mean([len(sub) for sub in subdomains]) if subdomains else 0
    
    # Average number of dots in subdomain
    features['average_number_of_dots_in_subdomain'] = np.mean([sub.count('.') for sub in subdomains]) if subdomains else 0
    
    # Average number of hyphens in subdomain
    features['average_number_of_hyphens_in_subdomain'] = np.mean([sub.count('-') for sub in subdomains]) if subdomains else 0
    
    # Having special characters in subdomain
    features['having_special_characters_in_subdomain'] = int(any(bool(re.search(f'[{special_chars}]', sub)) for sub in subdomains))
    
    # Number of special characters in subdomain
    features['number_of_special_characters_in_subdomain'] = sum(len(re.findall(f'[{special_chars}]', sub)) for sub in subdomains)
    
    # Having digits in subdomain
    features['having_digits_in_subdomain'] = int(any(any(c.isdigit() for c in sub) for sub in subdomains))
    
    # Number of digits in subdomain
    features['number_of_digits_in_subdomain'] = sum(sum(c.isdigit() for c in sub) for sub in subdomains)
    
    # Having repeated digits in subdomain
    features['having_repeated_digits_in_subdomain'] = int(any(bool(re.search(r'(\d)\1', sub)) for sub in subdomains))
    
    # Having path
    features['having_path'] = int(bool(path))
    
    # Path length
    features['path_length'] = len(path)
    
    # Having query
    features['having_query'] = int(bool(query))
    
    # Having fragment
    features['having_fragment'] = int(bool(fragment))
    
    # Having anchor
    features['having_anchor'] = int('#' in url)
    
    # Entropy of URL
    features['entropy_of_url'] = calculate_entropy(url)
    
    # Entropy of domain
    features['entropy_of_domain'] = calculate_entropy(domain)
    
    return features


