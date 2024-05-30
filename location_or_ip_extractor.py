import re

"""
ChatGPT: CHAT LOG: https://chatgpt.com/share/db2e3337-d356-47cc-a65d-97d4c26755d8
"""


# generate Local outlier factor model
m1 = """
May 09 08:46:47 staging-sdo-mobile-a1.sdo.local sdo-cloud-server[10696]: I <auth> Finishing auth D9WxH6rfREuFPvDhkZLBmZFk of 44087008-9595-4aa6-8fb9-0ec8e32e9fe8 for
048a6c2a ip: null via Authenticator/5.9.0.429 (iOS 17.4.1; iPhone14_4) - approved
"""

m2 = """
May 09 08:45:50 staging-sdo-mobile-a1.sdo.local sdo-cloud-server[10696]: I <auth> Finishing auth Y3doK4ZMQAevm9q9rGabbH8n of 44087008-9595-4aa6-8fb9-0ec8e32e9fe8 for
048a6c2a ip: 172.16.10.16 via Authenticator/5.9.0.429 (iOS 17.4.1; iPhone14_4) - approved
"""

m3 = """
May 09 08:43:43 staging-sdo-mobile-a1.sdo.local sdo-cloud-server[10696]: I <auth> Finishing auth qbXbTqkRqzeDb9RKzsw8tUAd of 27602cab-5d27-4f79-95e1-4cfb0a705108 for
0426400d at Pardes Hanna Karkur, Israel (84.108.134.210) via Authenticator/5.9.0.429 (iOS 17.4.1; iPhone14_4) - approved
"""

# Define regex patterns
ip_pattern = re.compile(r'ip: (null|\d{1,3}(?:\.\d{1,3}){3})')
location_pattern = re.compile(r'at (.*? \(\d{1,3}(?:\.\d{1,3}){3}\))')

def extract_location_or_ip(message):
    # Check for location pattern
    location_match = location_pattern.search(message)
    if location_match:
        return location_match.group(1)

    # Check for IP pattern
    ip_match = ip_pattern.search(message)
    if ip_match:
        return f"ip: {ip_match.group(1)}"

    return None




if __name__ == '__main__':
    # Extract and print results
    for idx, message in enumerate([m1, m2, m3], 1):
        result = extract_location_or_ip(message)
        print(f"Message {idx} result: {result}")