"""
Test inference script untuk format 3 produk baru
Hanya untuk testing format, bukan untuk training atau inference model
"""

def create_sample_prompt():
    """Buat contoh prompt dengan format baru"""
    prompt = """Kamu adalah host live TikTok Shop yang berbahasa santai. Berdasarkan detail 3 produk di bawah, buat copywriting yang menarik untuk masing-masing produk, rekomendasi bundling 2 produk dengan diskon, dan tentukan jam live yang optimal.

Contoh jawaban:
COPY1: Kaos oversize keren banget, cocok buat hangout santai!
COPY2: Kemeja flannel premium, bikin tampilan makin stylish!
COPY3: Celana jeans berkualitas, wajib punya buat koleksi!
BUNDLE: Kaos Oversize + Kemeja Flannel (diskon 20%)
TIME: 19:00-21:00 WIB

PRODUK 1:
Nama: Kaos Oversize Premium
Harga: Rp150,000
Terjual hari ini: 25 pcs

PRODUK 2:
Nama: Kemeja Flannel Stylish
Harga: Rp200,000
Terjual hari ini: 18 pcs

PRODUK 3:
Nama: Celana Jeans Slim Fit
Harga: Rp250,000
Terjual hari ini: 12 pcs
"""
    return prompt

def parse_expected_response(response):
    """Parse response untuk validasi format"""
    lines = response.strip().split('\n')
    parsed = {}
    
    for line in lines:
        if line.startswith('COPY1:'):
            parsed['copy1'] = line[6:].strip()
        elif line.startswith('COPY2:'):
            parsed['copy2'] = line[6:].strip()
        elif line.startswith('COPY3:'):
            parsed['copy3'] = line[6:].strip()
        elif line.startswith('BUNDLE:'):
            parsed['bundle'] = line[7:].strip()
        elif line.startswith('TIME:'):
            parsed['time'] = line[5:].strip()
    
    return parsed

def test_format():
    """Test format prompt dan response"""
    print("=== TEST FORMAT BARU ===\n")
    
    prompt = create_sample_prompt()
    print("PROMPT:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    expected_response = """COPY1: Kaos oversize premium ini wajib banget punya, nyaman dipakai kemana-mana!
COPY2: Kemeja flannel stylish bikin kamu keliatan kece dan trendy abis!
COPY3: Celana jeans slim fit berkualitas tinggi, cocok buat segala outfit!
BUNDLE: Celana Jeans Slim Fit + Kemeja Flannel Stylish (diskon 20%)
TIME: 19:00-21:00 WIB"""
    
    print("EXPECTED RESPONSE:")
    print(expected_response)
    print("\n" + "="*50 + "\n")
    
    parsed = parse_expected_response(expected_response)
    print("PARSED RESPONSE:")
    for key, value in parsed.items():
        print(f"{key.upper()}: {value}")
    
    print("\n=== FORMAT VALIDATION ===")
    required_fields = ['copy1', 'copy2', 'copy3', 'bundle', 'time']
    all_fields_present = all(field in parsed for field in required_fields)
    print(f"All required fields present: {all_fields_present}")
    
    if all_fields_present:
        print("✅ Format sesuai dengan requirement!")
    else:
        missing = [field for field in required_fields if field not in parsed]
        print(f"❌ Missing fields: {missing}")

if __name__ == "__main__":
    test_format()
