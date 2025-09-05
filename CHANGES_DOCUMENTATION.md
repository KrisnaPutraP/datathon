# Dokumentasi Perubahan Format Fine-Tuning untuk Multiple Product Bundling

## Ringkasan Perubahan

Model telah diubah dari format single product menjadi **3 products bundling format** dengan spesifikasi berikut:

### INPUT FORMAT (Prompt):
```
PRODUK 1:
Nama: [Nama Produk 1]
Harga: Rp[Harga]
Terjual hari ini: [Jumlah] pcs

PRODUK 2:
Nama: [Nama Produk 2]  
Harga: Rp[Harga]
Terjual hari ini: [Jumlah] pcs

PRODUK 3:
Nama: [Nama Produk 3]
Harga: Rp[Harga]
Terjual hari ini: [Jumlah] pcs
```

### OUTPUT FORMAT (Response):
```
COPY1: [Copywriting untuk produk 1]
COPY2: [Copywriting untuk produk 2]
COPY3: [Copywriting untuk produk 3]
BUNDLE: [Nama Produk A] + [Nama Produk B] (diskon [X]%)
TIME: [HH:MM-HH:MM] WIB
```

## File yang Diubah

### 1. `src/utils.py`
- **PROMPT_TEMPLATE**: Diubah untuk menerima 3 produk dengan nama, harga, dan jumlah terjual hari ini
- **build_full_dataset()**: Completely rewritten untuk:
  - Generate 3 produk per prompt (dari kategori yang sama/related)
  - Menggunakan jumlah penjualan random (5-50 pcs) sebagai "terjual hari ini"
  - Membuat 3 copywriting terpisah dari speech segments
  - Bundle 2 produk termahal dengan diskon random (15-25%)
  - Generate jam live optimal berdasarkan kategori dari data historical
  - **HOST dihapus** dari output sesuai permintaan

### 2. `train.py`
- **wrap_inst()**: Format instruction diubah menjadi "COPY1:, COPY2:, COPY3:, BUNDLE:, TIME:"
- **compute_metrics()**: Menambah evaluasi untuk 3 copywriting (acc_copy1, acc_copy2, acc_copy3)
- **Menghapus acc_host** dari metrics
- **BitsAndBytesConfig**: Diubah dari NF4 ke **FP4 quantization** untuk kualitas lebih baik dan kompatibilitas

### 3. `inference.py`
- **BitsAndBytesConfig**: Updated ke FP4 quantization untuk konsistensi dan menghindari error SCB

### 4. Dataset Generated
- **copy_train.jsonl** & **copy_valid.jsonl**: Format baru dengan 500 samples
- Setiap sample berisi 3 produk dengan copywriting dan bundling yang realistic

## Cara Menggunakan

### 1. Generate Dataset Baru
```bash
cd "path/to/project"
python train.py --prepare-data
```

### 2. Fine-tuning di Colab (dengan GPU)
```python
# Upload project ke Colab
# Install requirements
!pip install -r requirements.txt

# Run training
python train.py --train
```

### 3. Inference Format
Input 3 produk → Output 3 copywriting + 1 bundling + jam live optimal

## Keunggulan Format Baru

1. **Multiple Products**: Mendukung bundling yang lebih realistic
2. **Smart Bundling**: Memilih 2 produk termahal untuk bundling dengan profit margin optimal
3. **Historical Time Learning**: Jam live dipelajari dari data session yang sukses
4. **Sales-driven**: Menggunakan data penjualan hari ini sebagai context
5. **No Host Dependency**: Fokus pada produk dan copywriting, bukan host spesifik

## File Siap untuk Fine-tuning

✅ Dataset: `dataset/copy_train.jsonl`, `dataset/copy_valid.jsonl`
✅ Training Script: `train.py` 
✅ Config: `src/config.py`
✅ Utilities: `src/utils.py`

**Status**: Ready for fine-tuning di Colab dengan GPU!

## 🔧 Troubleshooting

### Error: `'Tensor' object has no attribute 'SCB'`
**Cause**: Konflik antara bitsandbytes version dengan 8-bit quantization dan device_map="auto"
**Solution**: Gunakan FP4 quantization (lebih baik dari NF4) dengan konfigurasi yang kompatibel:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16" if bf16_ok else "float16", 
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="fp4",  # FP4 > NF4 dalam kualitas
)
```

### Quantization Quality Ranking:
1. **FP16/BF16** (Best quality, high memory)
2. **INT8** (Good quality, medium memory) 
3. **FP4** (Good quality, low memory) ← **Current choice**
4. **NF4** (Poor quality, low memory)
