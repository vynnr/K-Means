import tkinter as tk
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# Data siswa
data_siswa = [
    {"NO": 1, "NAMA SISWA": "ADITYA YULIANSYAH", "NISN": "0064111253", "NIS": "2223-10-002", "MATA PELAJARAN": [80, 80, 76, 76, 74, 78, 77, 79, 77, 77, 77, 80, 79], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 2, "NAMA SISWA": "Afifah Nursa'diah", "NISN": "3061353699", "NIS": "2223-10-003", "MATA PELAJARAN": [85, 86, 84, 85, 81, 81, 81, 85, 88, 85, 85, 86, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 3, "NAMA SISWA": "Alika Refa Aulia", "NISN": "0077256348", "NIS": "2223-10-005", "MATA PELAJARAN": [84, 84, 83, 83, 80, 80, 81, 80, 80, 80, 80, 85, 80], "Ketidakhadiran": [0, 1, 2]},
    {"NO": 4, "NAMA SISWA": "ASTI RAMADHANI", "NISN": "3063758124", "NIS": "2223-10-006", "MATA PELAJARAN": [85, 84, 83, 82, 80, 80, 80, 80, 80, 81, 80, 85, 79], "Ketidakhadiran": [0, 0, 1]},
    {"NO": 5, "NAMA SISWA": "BILLY ARDIANSYAH", "NISN": "0053268131", "NIS": "2223-10-008", "MATA PELAJARAN": [85, 84, 82, 82, 79, 81, 82, 80, 81, 80, 79, 84, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 6, "NAMA SISWA": "Dede Mulyadi", "NISN": "0059745566", "NIS": "2223-10-009", "MATA PELAJARAN": [84, 84, 76, 79, 75, 80, 79, 79, 77, 77, 77, 82, 76], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 7, "NAMA SISWA": "Eva Rohaniyah", "NISN": "3075499160", "NIS": "2223-10-010", "MATA PELAJARAN": [85, 85, 83, 81, 80, 81, 81, 83, 80, 83, 80, 84, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 8, "NAMA SISWA": "Faiz Fadilah", "NISN": "0062412734", "NIS": "2223-10-011", "MATA PELAJARAN": [84, 84, 83, 80, 80, 81, 81, 79, 77, 79, 79, 84, 79], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 9, "NAMA SISWA": "FAJAR MIFTAHUL KHUNI", "NISN": "0066112170", "NIS": "2223-10-012", "MATA PELAJARAN": [85, 86, 80, 83, 80, 79, 82, 83, 80, 80, 80, 85, 80], "Ketidakhadiran": [0, 0, 1]},
    {"NO": 10, "NAMA SISWA": "FARHAN FAISAL", "NISN": "0079477948", "NIS": "2324-11-064", "MATA PELAJARAN": [81, 81, 77, 82, 79, 80, 80, 79, 77, 78, 78, 85, 79], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 11, "NAMA SISWA": "Farid Muhammad Rizki", "NISN": "0078653812", "NIS": "2223-10-062", "MATA PELAJARAN": [80, 80, 76, 79, 74, 80, 78, 77, 77, 77, 77, 82, 69], "Ketidakhadiran": [0, 0, 5]},
    {"NO": 12, "NAMA SISWA": "Hasbi Abdul Ghani", "NISN": "0079783073", "NIS": "2223-10-015", "MATA PELAJARAN": [84, 84, 82, 80, 74, 81, 82, 79, 77, 79, 77, 83, 79], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 13, "NAMA SISWA": "Hisnu Munawar", "NISN": "0065634495", "NIS": "2223-10-016", "MATA PELAJARAN": [85, 84, 83, 79, 74, 81, 79, 77, 77, 77, 77, 82, 78], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 14, "NAMA SISWA": "KHAIDAR SYUJA MUSAPA", "NISN": "0063445436", "NIS": "2223-10-017", "MATA PELAJARAN": [84, 83, 76, 79, 75, 79, 79, 79, 77, 79, 79, 82, 77], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 15, "NAMA SISWA": "M. AJI DARMAWAN", "NISN": "0063913029", "NIS": "2223-10-018", "MATA PELAJARAN": [87, 87, 85, 85, 84, 85, 85, 85, 87, 85, 87, 87, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 16, "NAMA SISWA": "Malla Hanjani", "NISN": "3076800824", "NIS": "2223-10-019", "MATA PELAJARAN": [85, 85, 83, 79, 80, 81, 81, 81, 81, 81, 80, 83, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 17, "NAMA SISWA": "MARIYAH ULFA", "NISN": "0077702250", "NIS": "2223-10-020", "MATA PELAJARAN": [88, 88, 83, 81, 80, 81, 81, 81, 81, 81, 80, 84, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 18, "NAMA SISWA": "Muhammad Rizki Agustin", "NISN": "3072351398", "NIS": "2223-10-031", "MATA PELAJARAN": [85, 85, 83, 81, 80, 79, 81, 80, 81, 80, 80, 85, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 19, "NAMA SISWA": "NAILA NANDA SAVIRA", "NISN": "0067137001", "NIS": "2223-10-021", "MATA PELAJARAN": [85, 85, 83, 82, 80, 82, 81, 81, 80, 81, 80, 85, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 20, "NAMA SISWA": "NAILA SEPRIYANTI", "NISN": "0064027324", "NIS": "2223-10-022", "MATA PELAJARAN": [85, 85, 85, 82, 83, 81, 83, 85, 85, 85, 85, 86, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 21, "NAMA SISWA": "Nova Aulia", "NISN": "0066704863", "NIS": "2223-10-023", "MATA PELAJARAN": [85, 85, 76, 80, 74, 80, 77, 77, 77, 77, 77, 80, 77], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 22, "NAMA SISWA": "Nur Saidah", "NISN": "3066077340", "NIS": "2223-10-024", "MATA PELAJARAN": [85, 85, 83, 82, 80, 81, 81, 80, 80, 80, 80, 86, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 23, "NAMA SISWA": "Rafli Fadillah", "NISN": "0061952904", "NIS": "2223-10-057", "MATA PELAJARAN": [80, 80, 76, 77, 74, 79, 79, 77, 77, 77, 77, 81, 75], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 24, "NAMA SISWA": "RIZQI KHAIDAR MIFTAHUL HUDA", "NISN": "0061328464", "NIS": "2223-10-065", "MATA PELAJARAN": [85, 86, 83, 84, 84, 81, 80, 90, 88, 86, 89, 86, 79], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 25, "NAMA SISWA": "Rizqiqa Hanifa", "NISN": "0069848554", "NIS": "2223-10-029", "MATA PELAJARAN": [85, 85, 81, 79, 76, 80, 77, 77, 77, 77, 77, 82, 78], "Ketidakhadiran": [0, 0, 3]},
    {"NO": 26, "NAMA SISWA": "ROBI AHMAD ASAUKI", "NISN": "0072598198", "NIS": "2223-10-025", "MATA PELAJARAN": [87, 85, 83, 79, 75, 79, 78, 77, 77, 78, 77, 83, 80], "Ketidakhadiran": [0, 0, 2]},
    {"NO": 27, "NAMA SISWA": "Shofiah Nurul Amin", "NISN": "3077189714", "NIS": "2223-10-026", "MATA PELAJARAN": [88, 88, 85, 86, 81, 82, 83, 85, 85, 85, 85, 88, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 28, "NAMA SISWA": "Siti Marwah", "NISN": "0069831165", "NIS": "2223-10-027", "MATA PELAJARAN": [88, 88, 85, 85, 84, 82, 83, 87, 87, 85, 87, 87, 80], "Ketidakhadiran": [0, 0, 0]},
    {"NO": 29, "NAMA SISWA": "Tia Lestari", "NISN": "3063013873", "NIS": "2223-10-028", "MATA PELAJARAN": [88, 88, 84, 85, 82, 82, 82, 85, 87, 87, 88, 86, 80], "Ketidakhadiran": [0, 0, 0]},
]

# Fungsi untuk menghitung tingkat kelulusan
def hitung_tingkat_kelulusan(data_siswa):
    nilai_rata_rata = [np.mean(siswa["MATA PELAJARAN"]) for siswa in data_siswa]
    nilai_minimal_lulus = 75  # Nilai minimal untuk lulus
    jumlah_lulus = sum(1 for nilai in nilai_rata_rata if nilai >= nilai_minimal_lulus)
    jumlah_tidak_lulus = len(data_siswa) - jumlah_lulus
    return jumlah_lulus, jumlah_tidak_lulus

# Fungsi untuk melakukan clustering
def clustering_data(data_siswa, k):
    X = np.array([siswa["MATA PELAJARAN"] for siswa in data_siswa])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    return labels

# Fungsi untuk menampilkan confusion matrix
def tampilkan_confusion_matrix(labels, data_siswa):
    cluster_count = len(set(labels))
    true_labels = [siswa["Ketidakhadiran"][0] for siswa in data_siswa]
    cm = confusion_matrix(true_labels, labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(cluster_count)
    plt.xticks(tick_marks, range(cluster_count))
    plt.yticks(tick_marks, range(cluster_count))
    plt.xlabel('Cluster Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

# Fungsi untuk melakukan proses analisis dan menyimpan ke file
def proses_analisis_dan_simpan():
    jumlah_lulus, jumlah_tidak_lulus = hitung_tingkat_kelulusan(data_siswa)
    messagebox.showinfo("Hasil Analisis", f"Jumlah siswa yang lulus: {jumlah_lulus}\nJumlah siswa yang tidak lulus: {jumlah_tidak_lulus}")
    
    # Clustering data
    k = 3  # Jumlah cluster yang diinginkan
    labels = clustering_data(data_siswa, k)
    
    # Tampilkan dan simpan confusion matrix
    tampilkan_confusion_matrix(labels, data_siswa)
    
    # Simpan hasil analisis ke file
    simpan_ke_file(labels)

# Fungsi untuk menyimpan hasil analisis ke file
def simpan_ke_file(labels):
    filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if filename:
        with open(filename, 'w') as file:
            file.write("Hasil Clustering Siswa:\n")
            for i, siswa in enumerate(data_siswa):
                file.write(f"{siswa['NAMA SISWA']}: Cluster {labels[i]}\n")
        messagebox.showinfo("File Disimpan", "Hasil analisis berhasil disimpan ke dalam file.")

# GUI menggunakan Tkinter
root = tk.Tk()
root.title("Analisis Kelulusan dan Clustering Siswa")

# Label judul
label_judul = tk.Label(root, text="Analisis Kelulusan dan Clustering Siswa", font=("Arial", 14))
label_judul.pack(pady=10)

# Tombol untuk proses analisis dan simpan
tombol_proses = tk.Button(root, text="Proses Analisis dan Simpan", command=proses_analisis_dan_simpan, padx=10, pady=5)
tombol_proses.pack(pady=10)

# Menjalankan main loop GUI
root.mainloop()
