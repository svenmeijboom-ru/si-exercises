import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imageio

def quantize_image(image, palette):
    h, w, _ = image.shape
    flat_img = image.reshape(-1, 3)
    distances = np.sqrt(((flat_img[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2))
    nearest_idx = np.argmin(distances, axis=1)
    quantized_flat = palette[nearest_idx]
    quantized_image = quantized_flat.reshape(h, w, 3)
    return quantized_image

def compute_fitness(palette, image):
    quantized = quantize_image(image, palette)
    error = np.sum((image - quantized) ** 2)
    return error

def pso_color_quantization(image, n_clusters=16, n_particles=30, n_iterations=50, w=0.5, c1=1.5, c2=1.5):
    image = image.astype(np.float32)
    
    particles = np.random.uniform(0, 255, (n_particles, n_clusters, 3))
    velocities = np.random.uniform(-10, 10, (n_particles, n_clusters, 3))
    
    pbest_positions = np.copy(particles)
    pbest_fitness = np.array([compute_fitness(particles[i], image) for i in range(n_particles)])
    
    gbest_index = np.argmin(pbest_fitness)
    gbest_position = np.copy(pbest_positions[gbest_index])
    gbest_fitness = pbest_fitness[gbest_index]
    
    gbest_history = []
    palette_history = []
    
    for it in range(n_iterations):
        for i in range(n_particles):
            r1 = np.random.rand(n_clusters, 3)
            r2 = np.random.rand(n_clusters, 3)

            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - particles[i]) +
                             c2 * r2 * (gbest_position - particles[i]))
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, 255)
            
            fitness = compute_fitness(particles[i], image)
            if fitness < pbest_fitness[i]:
                pbest_positions[i] = np.copy(particles[i])
                pbest_fitness[i] = fitness

                if fitness < gbest_fitness:
                    gbest_position = np.copy(particles[i])
                    gbest_fitness = fitness
        
        gbest_history.append(gbest_fitness)
        palette_history.append(np.copy(gbest_position))
        
        if (it + 1) % 10 == 0 or it == 0:
            quantized = quantize_image(image, gbest_position)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            palette_img = np.zeros((50, n_clusters * 50, 3), dtype=np.uint8)
            for j in range(n_clusters):
                palette_img[:, j*50:(j+1)*50, :] = np.uint8(np.tile(gbest_position[j], (50, 50, 1)))
            plt.imshow(palette_img)
            plt.title(f'Iteration {it + 1} - Best Palette')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(np.uint8(quantized))
            plt.title(f'Iteration {it + 1} - Quantized Image')
            plt.axis('off')
            plt.show()
            print(f"Iteration {it + 1}, Global Best Fitness: {gbest_fitness:.2f}")
    
    return gbest_position, gbest_history, palette_history

def kmeans_color_quantization(image, n_clusters=16):
    h, w, _ = image.shape
    flat_img = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(flat_img)
    palette = kmeans.cluster_centers_
    labels = kmeans.labels_
    quantized_flat = palette[labels]
    quantized_image = quantized_flat.reshape(h, w, 3)
    return palette, quantized_image

def main():
    image = imageio.imread("image.png")
    
    n_clusters = 4

    n_particles = 30
    n_iterations = 50
    w = 0.73      # Inertia weight.
    c1 = 1.5     # Cognitive acceleration coefficient.
    c2 = 1.5     # Social acceleration coefficient.
    
    print("Running PSO-based color quantization...")
    best_palette, history, palette_history = pso_color_quantization(image, n_clusters, n_particles, n_iterations, w, c1, c2)
    quantized_image_pso = quantize_image(image, best_palette)
    
    print("Running K-means based color quantization...")
    kmeans_palette, quantized_image_kmeans = kmeans_color_quantization(image, n_clusters)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(3, 2, 3)
    palette_img = np.zeros((50, n_clusters * 50, 3), dtype=np.uint8)
    for j in range(n_clusters):
        palette_img[:, j*50:(j+1)*50, :] = np.uint8(np.tile(best_palette[j], (50, 50, 1)))
    plt.imshow(palette_img)
    plt.title("PSO Best Palette")
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    plt.imshow(np.uint8(quantized_image_pso))
    plt.title("Quantized Image (PSO)")
    plt.axis('off')

    plt.subplot(3, 2, 5)
    palette_img = np.zeros((50, n_clusters * 50, 3), dtype=np.uint8)
    for j in range(n_clusters):
        palette_img[:, j*50:(j+1)*50, :] = np.uint8(np.tile(kmeans_palette[j], (50, 50, 1)))
    plt.imshow(palette_img)
    plt.title("K-Means Best Palette")
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.imshow(np.uint8(quantized_image_kmeans))
    plt.title("Quantized Image (K-means)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
