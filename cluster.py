import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os

# Data mentah (nama, uts, uas)
data = {
    "Nama": [
        "ADITYA YULIANSYAH", "Afifah Nursa'diah", "Alika Refa Aulia", "ASTI RAMADHANI",
        "BILLY ARDIANSYAH", "Dede Mulyadi", "Eva Rohaniyah", "Faiz Fadilah",
        "FAJAR MIFTAHUL KHUNI", "FARHAN FAISAL", "Farid Muhammad Rizki", "Hasbi Abdul Ghani",
        "Hisnu Munawar", "KHAIDAR SYUJA MUSAPA", "M. AJI DARMAWAN", "Malla Hanjani",
        "MARIYAH ULFA", "Muhammad Rizki Agustin", "NAILA NANDA SAVIRA", "NAILA SEPRIYANTI",
        "Nova Aulia", "Nur Saidah", "Rafli Fadillah", "RIZQI KHAIDAR MIFTAHUL HUDA",
        "Rizqiqa Hanifa", "ROBI AHMAD ASAUKI", "Shofiah Nurul Amin", "Siti Marwah", "Tia Lestari"
    ],
    "UTS": [
        80, 85, 84, 85, 85, 84, 85, 84, 85, 81, 80, 84, 85, 84, 87, 85, 88, 85, 85, 85,
        85, 85, 80, 85, 85, 87, 88, 88, 88, 88
    ],
    "UAS": [
        80, 86, 84, 84, 84, 84, 85, 84, 86, 81, 80, 84, 84, 83, 87, 85, 88, 85, 85, 85,
        85, 85, 80, 86, 85, 85, 88, 88, 88, 88
    ]
}

# Pastikan semua array memiliki panjang yang sama
panjang_nama = len(data["Nama"])
panjang_uts = len(data["UTS"])
panjang_uas = len(data["UAS"])

if not (panjang_nama == panjang_uts == panjang_uas):
    # Jika panjangnya tidak sama, cari panjang minimum
    panjang_minimal = min(panjang_nama, panjang_uts, panjang_uas)
    data["Nama"] = data["Nama"][:panjang_minimal]
    data["UTS"] = data["UTS"][:panjang_minimal]
    data["UAS"] = data["UAS"][:panjang_minimal]

# Konversi data ke dalam DataFrame
df = pd.DataFrame(data)

# Fungsi untuk menampilkan scatter plot
def tampilkan_scatter_plot():
    plt.figure(figsize=(8, 6))
    plt.scatter(df['UTS'], df['UAS'], s=50, cmap='viridis')
    plt.title('Scatter Plot: Nilai UTS vs UAS')
    plt.xlabel('Nilai UTS')
    plt.ylabel('Nilai UAS')
    plt.grid(True)
    plt.tight_layout()

    # Menyimpan plot sebagai file gambar
    file_path = os.path.join(os.getcwd(), "scatter_plot.png")
    plt.savefig(file_path)
    plt.show()

    # Menampilkan penjelasan hasil
    pesan = (
        "Scatter plot menunjukkan distribusi data berdasarkan nilai UTS dan UAS. "
        "Posisi titik-titik pada plot menggambarkan hubungan antara nilai UTS dan UAS. "
        "Titik-titik yang berkumpul dekat menunjukkan kemiripan nilai antara siswa-siswa tertentu."
    )
    messagebox.showinfo("Hasil Scatter Plot", pesan)

# Fungsi untuk melakukan clustering menggunakan K-means
def lakukan_clustering(jumlah_cluster):
    data = df[['UTS', 'UAS']]
    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return labels, centroids

# Fungsi untuk menampilkan confusion matrix
def tampilkan_confusion_matrix(labels):
    # Di sini contoh confusion matrix adalah confusion matrix dari clustering
    y_asli = np.zeros(len(df))  # Anggap semua data sebagai satu kelas
    cm = confusion_matrix(y_asli, labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(labels)))
    plt.xticks(tick_marks, np.unique(labels))
    plt.yticks(tick_marks, np.unique(labels))

    plt.tight_layout()
    plt.ylabel('Label Asli')
    plt.xlabel('Label Prediksi')

    # Menyimpan plot sebagai file gambar
    file_path = os.path.join(os.getcwd(), "confusion_matrix.png")
    plt.savefig(file_path)
    plt.show()

    # Menampilkan penjelasan hasil
    pesan = (
        "Confusion matrix menunjukkan hasil dari clustering berdasarkan nilai UTS dan UAS. "
        "Ini membandingkan label asli dengan label yang diprediksi oleh algoritma clustering."
    )
    messagebox.showinfo("Hasil Confusion Matrix", pesan)

# Fungsi untuk membuat Elbow Plot dan menentukan jumlah cluster optimal
def tampilkan_elbow_plot():
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df[['UTS', 'UAS']])
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Plot untuk Penentuan Jumlah Cluster')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.tight_layout()

    # Menyimpan plot sebagai file gambar
    file_path = os.path.join(os.getcwd(), "elbow_plot.png")
    plt.savefig(file_path)
    plt.show()

    # Menampilkan penjelasan hasil
    pesan = (
        "Elbow plot digunakan untuk menentukan jumlah cluster yang optimal. "
        "Plot menunjukkan penurunan inertia saat jumlah cluster meningkat. "
        "Pada titik 'elbow' (siku), penurunan inertia menjadi lebih lambat, menandakan jumlah cluster yang optimal."
    )
    messagebox.showinfo("Hasil Elbow Plot", pesan)

def buat_gui():
    # Membuat instance dari Tkinter
    root = tk.Tk()
    root.title("Aplikasi Clustering dan Confusion Matrix")

    # Mengatur ukuran dan posisi window
    root.geometry("600x400+200+200")

    # Label untuk scatter plot
    label_scatter = tk.Label(root, text="Tekan tombol untuk menampilkan scatter plot:")
    label_scatter.pack(pady=10)

    # Tombol untuk tampilkan scatter plot
    tombol_scatter = tk.Button(root, text="Tampilkan Scatter Plot", command=tampilkan_scatter_plot)
    tombol_scatter.pack()

    # Label untuk Elbow Plot
    label_elbow = tk.Label(root, text="Tekan tombol untuk menampilkan Elbow Plot:")
    label_elbow.pack(pady=10)

    # Tombol untuk tampilkan Elbow Plot
    tombol_elbow = tk.Button(root, text="Tampilkan Elbow Plot", command=tampilkan_elbow_plot)
    tombol_elbow.pack()

    # Fungsi saat tombol "Lakukan Clustering" ditekan
    def tombol_clustering_ditekan():
        try:
            jumlah_cluster = int(entry_cluster.get())
            labels, centroids = lakukan_clustering(jumlah_cluster)
            messagebox.showinfo("Clustering Selesai", "Clustering selesai!")
            tampilkan_confusion_matrix(labels)
        except ValueError:
            messagebox.showerror("Error", "Masukkan jumlah cluster dalam format yang benar!")

    # Label untuk clustering
    label_cluster = tk.Label(root, text="Atau masukkan jumlah cluster untuk K-means:")
    label_cluster.pack(pady=10)

    # Entry untuk input jumlah cluster
    entry_cluster = tk.Entry(root, width=10)
    entry_cluster.pack()

    # Tombol untuk lakukan clustering
    tombol_clustering = tk.Button(root, text="Lakukan Clustering", command=tombol_clustering_ditekan)
    tombol_clustering.pack(pady=10)

    # Menjalankan main loop
    root.mainloop()

# Memanggil fungsi untuk membuat GUI
buat_gui()
