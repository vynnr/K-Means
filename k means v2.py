import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import numpy as np

class KMeansGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering")
        
        self.num_clusters_label = tk.Label(self.root, text="Number of Clusters:")
        self.num_clusters_label.pack()
        
        self.num_clusters_entry = tk.Entry(self.root)
        self.num_clusters_entry.pack()
        
        self.generate_data_button = tk.Button(self.root, text="Generate Data", command=self.generate_data)
        self.generate_data_button.pack()
        
        self.cluster_button = tk.Button(self.root, text="Cluster", command=self.cluster)
        self.cluster_button.pack()
        
        self.confusion_matrix_button = tk.Button(self.root, text="Confusion Matrix", command=self.show_confusion_matrix)
        self.confusion_matrix_button.pack()
        
        self.canvas = None
        self.data = None
        self.labels = None
        self.cluster_centers = None
        
    def generate_data(self):
        try:
            num_clusters = int(self.num_clusters_entry.get())
            self.data, self.labels = make_blobs(n_samples=300, centers=num_clusters, random_state=42)
            
            plt.figure(figsize=(6, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis')
            plt.title('Generated Data')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of clusters.")

    def cluster(self):
        try:
            num_clusters = int(self.num_clusters_entry.get())
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(self.data)
            self.labels = kmeans.predict(self.data)
            self.cluster_centers = kmeans.cluster_centers_
            
            plt.figure(figsize=(6, 6))
            plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, cmap='viridis')
            plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], marker='x', s=200, c='red')
            plt.title('Clustered Data')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
        except ValueError:
            messagebox.showerror("Error", "Please generate data first.")
            
    def show_confusion_matrix(self):
        if self.labels is not None:
            true_labels = np.argmax(self.data, axis=1)
            cm = confusion_matrix(true_labels, self.labels)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Cluster label')
            plt.show()
        else:
            messagebox.showerror("Error", "Please cluster data first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansGUI(root)
    root.mainloop()
