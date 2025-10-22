import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- FUNGSI UNTUK MENGHITUNG DAN MENAMPILKAN HISTOGRAM ---
def plot_histograms(image):
    """
    Fungsi ini akan menghitung dan menampilkan histogram untuk 
    citra grayscale, hasil equalization, biner, dan berwarna.
    """
    
    # --- PROSES UNTUK CITRA GRAYSCALE ---
    st.subheader("Analisis Histogram Grayscale (Asli)")
    
    # Konversi citra ke grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Tampilkan citra grayscale
    st.image(gray_image, caption='Citra Grayscale Asli', width=300)

    # Hitung histogram normal untuk grayscale
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Hitung histogram ternormalisasi
    total_pixels_gray = gray_image.shape[0] * gray_image.shape[1]
    hist_gray_normalized = hist_gray / total_pixels_gray
    
    # Plotting histogram grayscale
    fig_gray, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot histogram normal
    ax1.set_title("Histogram Grayscale Asli (Normal)")
    ax1.set_xlabel("Intensitas Piksel")
    ax1.set_ylabel("Jumlah Piksel")
    ax1.plot(hist_gray, color='black')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, 256])
    
    # Plot histogram ternormalisasi
    ax2.set_title("Histogram Grayscale Asli (Ternormalisasi)")
    ax2.set_xlabel("Intensitas Piksel")
    ax2.set_ylabel("Probabilitas")
    ax2.plot(hist_gray_normalized, color='black')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim([0, 256])
    
    st.pyplot(fig_gray)

    # --- (BARU) PROSES UNTUK HISTOGRAM EQUALIZATION ---
    st.subheader("Analisis Histogram Grayscale (Hasil Equalization)")
    st.write("Histogram Equalization (HE) adalah teknik untuk meningkatkan kontras citra dengan meratakan distribusi histogram.")

    # Lakukan equalization pada gambar grayscale
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Tampilkan gambar hasil equalization
    st.image(equalized_image, caption='Citra Grayscale (Telah Diekualisasi)', width=300)

    # Hitung histogram normal untuk gambar yang telah diekualisasi
    hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    
    # Hitung histogram ternormalisasi
    # Kita bisa pakai ulang total_pixels_gray dari section sebelumnya
    hist_eq_normalized = hist_eq / total_pixels_gray
    
    # Plotting histogram hasil equalization
    fig_eq, (ax_eq1, ax_eq2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot histogram normal hasil equalization
    ax_eq1.set_title("Histogram Hasil Equalization (Normal)")
    ax_eq1.set_xlabel("Intensitas Piksel")
    ax_eq1.set_ylabel("Jumlah Piksel")
    ax_eq1.plot(hist_eq, color='blue') # Pakai warna biru agar beda
    ax_eq1.grid(True, linestyle='--', alpha=0.6)
    ax_eq1.set_xlim([0, 256])
    
    # Plot histogram ternormalisasi hasil equalization
    ax_eq2.set_title("Histogram Hasil Equalization (Ternormalisasi)")
    ax_eq2.set_xlabel("Intensitas Piksel")
    ax_eq2.set_ylabel("Probabilitas")
    ax_eq2.plot(hist_eq_normalized, color='blue')
    ax_eq2.grid(True, linestyle='--', alpha=0.6)
    ax_eq2.set_xlim([0, 256])
    
    st.pyplot(fig_eq)

    # --- PROSES UNTUK CITRA BINER (THRESHOLDING) ---
    st.subheader("Analisis Histogram Biner (Otsu's Thresholding)")
    st.write("Mengubah citra grayscale menjadi biner (hitam/putih) menggunakan nilai ambang batas (threshold) yang ditemukan otomatis oleh metode Otsu.")

    # Terapkan Otsu's Thresholding
    ret_otsu, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    st.write(f"Nilai ambang batas (threshold) yang terdeteksi oleh Otsu: **{ret_otsu:.2f}**")
    st.image(binary_image, caption='Citra Biner (Hasil Thresholding)', width=300)

    # Hitung jumlah piksel hitam dan putih
    total_pixels_bin = binary_image.size
    white_pixels = cv2.countNonZero(binary_image)
    black_pixels = total_pixels_bin - white_pixels
    
    # Data untuk plot
    labels = ['Hitam (0)', 'Putih (255)']
    counts = [black_pixels, white_pixels]
    probabilities = [black_pixels / total_pixels_bin, white_pixels / total_pixels_bin]

    # Plotting histogram biner (lebih cocok pakai bar chart)
    fig_bin, (ax_bin1, ax_bin2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram normal
    ax_bin1.set_title("Histogram Biner Normal")
    ax_bin1.set_ylabel("Jumlah Piksel")
    ax_bin1.bar(labels, counts, color=['black', 'lightgray'], edgecolor='black')
    
    # Plot histogram ternormalisasi
    ax_bin2.set_title("Histogram Biner Ternormalisasi")
    ax_bin2.set_ylabel("Probabilitas")
    ax_bin2.bar(labels, probabilities, color=['black', 'lightgray'], edgecolor='black')
    ax_bin2.set_ylim([0, 1])
    
    st.pyplot(fig_bin)

    # --- PROSES UNTUK CITRA BERWARNA (RGB) ---
    st.subheader("Analisis Histogram Warna (RGB)")
    st.write("Menampilkan histogram untuk masing-masing channel warna: Merah (Red), Hijau (Green), dan Biru (Blue).")
    
    # Pisahkan channel warna
    colors = ('r', 'g', 'b')
    channel_names = ('Red', 'Green', 'Blue')
    
    # Buat figure untuk histogram normal dan ternormalisasi
    fig_color_normal, ax_normal = plt.subplots(figsize=(8, 5))
    fig_color_normalized, ax_normalized = plt.subplots(figsize=(8, 5))

    ax_normal.set_title("Histogram Warna Normal")
    ax_normal.set_xlabel("Intensitas Piksel")
    ax_normal.set_ylabel("Jumlah Piksel")
    
    ax_normalized.set_title("Histogram Warna Ternormalisasi")
    ax_normalized.set_xlabel("Intensitas Piksel")
    ax_normalized.set_ylabel("Probabilitas")

    total_pixels_color = image.shape[0] * image.shape[1]

    for i, color in enumerate(colors):
        # Hitung histogram untuk setiap channel
        hist_color = cv2.calcHist([image], [i], None, [256], [0, 256])
        
        # Hitung histogram ternormalisasi
        hist_color_normalized = hist_color / total_pixels_color
        
        # Plot di axes masing-masing
        ax_normal.plot(hist_color, color=color, label=f'Channel {channel_names[i]}')
        ax_normalized.plot(hist_color_normalized, color=color, label=f'Channel {channel_names[i]}')

    ax_normal.legend()
    ax_normal.grid(True, linestyle='--', alpha=0.6)
    ax_normal.set_xlim([0, 256])
    
    ax_normalized.legend()
    ax_normalized.grid(True, linestyle='--', alpha=0.6)
    ax_normalized.set_xlim([0, 256])

    st.pyplot(fig_color_normal)
    st.pyplot(fig_color_normalized)


# --- TAMPILAN UTAMA APLIKASI STREAMLIT ---

st.set_page_config(layout="wide")
st.title("Aplikasi Analisis Histogram Citra")
st.write("Dibuat untuk tugas mata kuliah Pengolahan Citra.")
st.write("Unggah sebuah gambar untuk melihat berbagai jenis histogram.")

# Widget untuk upload file
uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca file gambar yang diunggah sebagai array byte
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode array byte menjadi format gambar OpenCV
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Konversi dari BGR ke RGB 
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    st.header("Citra Asli")
    # Perbaikan: Menghapus parameter 'use_column_width' yang deprecated
    st.image(image_rgb, caption='Gambar yang diunggah', width=400)
    
    st.markdown("---") # Garis pemisah
    
    # Panggil fungsi untuk memproses dan menampilkan histogram
    plot_histograms(image_rgb)
else:
    st.info("Silakan unggah sebuah gambar untuk memulai analisis.")
