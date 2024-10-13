import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import math
import csv
import os

# 1. Data Preprocessing
def preprocess_image(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def rgb_to_grayscale(image):
    return tf.image.rgb_to_grayscale(image)

def load_and_preprocess_data(split='train', batch_size=32, num_samples=4000):
    """
    Load and preprocess CIFAR-10 dataset.

    Args:
        split (str): Dataset split to load ('train' or 'test').
        batch_size (int): Batch size.
        num_samples (int): Number of samples to take from the split.

    Returns:
        tf.data.Dataset: Preprocessed and batched dataset.
    """
    # Load CIFAR-10 dataset
    dataset, info = tfds.load('cifar10', split=split, with_info=True, as_supervised=True)
    dataset = dataset.take(num_samples)

    # Preprocess and batch the data
    total_samples = num_samples
    with tqdm(total=total_samples, desc=f"Preprocessing {split} data") as pbar:
        def preprocess_and_update(img, label):
            pbar.update(1)
            return preprocess_image(img), preprocess_image(img)

        dataset = dataset.map(preprocess_and_update, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda img_color, img_color2: (rgb_to_grayscale(img_color), img_color2), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).shuffle(4000).prefetch(tf.data.AUTOTUNE)

    return dataset

# 2. Model Definitions
def build_generator():
    inputs = keras.Input(shape=(32, 32, 1))
    
    # Encoder
    e1 = keras.layers.Conv2D(64, 4, strides=2, padding='same')(inputs)  # 16x16x64
    e1 = keras.layers.LeakyReLU(0.2)(e1)
    
    e2 = keras.layers.Conv2D(128, 4, strides=2, padding='same')(e1)    # 8x8x128
    e2 = keras.layers.BatchNormalization()(e2)
    e2 = keras.layers.LeakyReLU(0.2)(e2)
    
    # Bridge
    b = keras.layers.Conv2D(256, 4, strides=1, padding='same')(e2)     # 8x8x256
    b = keras.layers.BatchNormalization()(b)
    b = keras.layers.LeakyReLU(0.2)(b)
    
    # Decoder
    d2 = keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same')(b)  # 16x16x128
    d2 = keras.layers.BatchNormalization()(d2)
    d2 = keras.layers.ReLU()(d2)
    d2 = keras.layers.Concatenate()([d2, e1])                               # 16x16x192
    
    # Optionally, add a Conv2D layer to reduce channels
    d2 = keras.layers.Conv2D(128, 3, padding='same')(d2)                   # 16x16x128
    d2 = keras.layers.BatchNormalization()(d2)
    d2 = keras.layers.ReLU()(d2)
    
    d1 = keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d2)   # 32x32x64
    d1 = keras.layers.BatchNormalization()(d1)
    d1 = keras.layers.ReLU()(d1)
    d1 = keras.layers.Concatenate()([d1, inputs])                            # 32x32x65
    
    # Optionally, add a Conv2D layer to reduce channels
    d1 = keras.layers.Conv2D(64, 3, padding='same')(d1)                     # 32x32x64
    d1 = keras.layers.BatchNormalization()(d1)
    d1 = keras.layers.ReLU()(d1)
    
    outputs = keras.layers.Conv2D(3, 4, strides=1, padding='same', activation='tanh')(d1)  # 32x32x3
    
    return keras.Model(inputs, outputs)

def build_discriminator():
    input_img = keras.Input(shape=(32, 32, 1))
    target_img = keras.Input(shape=(32, 32, 3))
    x = keras.layers.Concatenate()([input_img, target_img])
    
    x = keras.layers.Conv2D(64, 4, strides=2, padding='same')(x)  # 16x16x64
    x = keras.layers.LeakyReLU(0.2)(x)
    
    x = keras.layers.Conv2D(128, 4, strides=2, padding='same')(x)  # 8x8x128
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    
    x = keras.layers.Conv2D(256, 4, strides=2, padding='same')(x)  # 4x4x256
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    
    x = keras.layers.Conv2D(1, 4, strides=1, padding='same')(x)    # 4x4x1
    
    return keras.Model([input_img, target_img], x)

# 3. Loss Functions
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_loss = gan_loss + (100 * l1_loss)
    return total_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_loss = real_loss + generated_loss
    return total_loss

# 4. Training Step
@tf.function
def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_total_loss, disc_loss

# 5. Image Generation and Saving
def generate_and_save_images(model, epoch, test_input, output_dir='output_4000', max_images=16):
    predictions = model(test_input, training=False)
    
    num_images = min(predictions.shape[0], max_images)
    grid_size = math.ceil(math.sqrt(num_images))
    
    fig = plt.figure(figsize=(12, 12))
    
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i+1)
        img = (predictions[i] * 0.5) + 0.5  # Rescale to [0,1]
        plt.imshow(img.numpy())
        plt.axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()

# 6. Evaluation Metrics
def calculate_metrics(real_images, generated_images, win_size=5):
    """
    Calculate the average SSIM and PSNR between real and generated images.

    Args:
        real_images (tf.Tensor): Batch of real images.
        generated_images (tf.Tensor): Batch of generated images.
        win_size (int): Window size for SSIM. Must be odd and <= min(image dimensions).

    Returns:
        tuple: (average SSIM, average PSNR)
    """
    ssim_scores = []
    psnr_scores = []
    
    real_images = (real_images * 0.5) + 0.5  # Rescale to [0,1]
    generated_images = (generated_images * 0.5) + 0.5  # Rescale to [0,1]
    
    real_images = tf.clip_by_value(real_images, 0.0, 1.0)
    generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)
    
    real_images = real_images.numpy()
    generated_images = generated_images.numpy()
    
    batch_size = real_images.shape[0]
    
    with tqdm(total=batch_size, desc="Calculating metrics") as pbar:
        for i in range(batch_size):
            real = (real_images[i] * 255).astype(np.uint8)
            generated = (generated_images[i] * 255).astype(np.uint8)
            
            # Ensure images have 3 channels
            if real.ndim == 3 and real.shape[-1] == 1:
                real = np.squeeze(real, axis=-1)
            if generated.ndim == 3 and generated.shape[-1] == 1:
                generated = np.squeeze(generated, axis=-1)
            
            # Debug: Print image shapes
            # print(f"Real image shape: {real.shape}, Generated image shape: {generated.shape}")
            
            # Calculate SSIM
            try:
                ssim_val = ssim(
                    real, 
                    generated, 
                    win_size=win_size, 
                    channel_axis=-1,
                    data_range=255
                )
            except ValueError as e:
                print(f"SSIM calculation error at index {i}: {e}")
                ssim_val = 0  # Assign a default value or handle as needed
            
            # Calculate PSNR
            psnr_val = psnr(
                real, 
                generated, 
                data_range=255
            )
            
            ssim_scores.append(ssim_val)
            psnr_scores.append(psnr_val)
            
            pbar.update(1)
    
    return np.mean(ssim_scores), np.mean(psnr_scores)

# 7. Plotting Functions
def plot_losses(csv_path, output_dir='output_4000'):
    epochs = []
    gen_losses = []
    disc_losses = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['Epoch']))
            gen_losses.append(float(row['Generator Loss']))
            disc_losses.append(float(row['Discriminator Loss']))
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, gen_losses, label='Generator Loss')
    plt.plot(epochs, disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'losses_plot.png'))
    plt.close()

def plot_metrics(csv_path, output_dir='output_4000'):
    epochs = []
    ssim_scores = []
    psnr_scores = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row['Epoch']))
            ssim_scores.append(float(row['SSIM']))
            psnr_scores.append(float(row['PSNR']))
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, ssim_scores, label='SSIM')
    plt.plot(epochs, psnr_scores, label='PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Score')
    plt.title('SSIM and PSNR per Evaluation Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.close()

# 8. Main Execution
def main():
    # Create a directory to save CSVs and images if not exists
    output_dir = 'output_4000'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV files
    losses_csv = os.path.join(output_dir, 'losses.csv')
    metrics_csv = os.path.join(output_dir, 'metrics.csv')
    
    with open(losses_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['Epoch', 'Generator Loss', 'Discriminator Loss'])
    
    with open(metrics_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['Epoch', 'SSIM', 'PSNR'])
    
    # Load and preprocess data
    train_dataset = load_and_preprocess_data(split='train', batch_size=32, num_samples=4000)
    test_dataset = load_and_preprocess_data(split='test', batch_size=32, num_samples=100)
    
    # Build models
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Define optimizers
    generator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # Define number of epochs
    epochs = 100  # Increased to 100 for better evaluation frequency
    
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        gen_losses = []
        disc_losses = []
        
        epoch_desc = f"Epoch {epoch+1}/{epochs}"
        for input_image, target in tqdm(train_dataset, desc=epoch_desc, leave=False):
            gen_total_loss, disc_loss = train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer)
            gen_losses.append(gen_total_loss)
            disc_losses.append(disc_loss)
        
        avg_gen_loss = tf.reduce_mean(gen_losses).numpy()
        avg_disc_loss = tf.reduce_mean(disc_losses).numpy()
        
        tqdm.write(f"Epoch {epoch+1}/{epochs} - Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")
        
        # Log to losses CSV
        with open(losses_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_gen_loss, avg_disc_loss])
        
        # Evaluate and log metrics every 10 epochs
        if (epoch) % 10 == 0:
            # Collect all test images
            test_grayscale = []
            test_color = []
            for batch in test_dataset:
                grayscale, color = batch
                test_grayscale.append(grayscale)
                test_color.append(color)
            
            test_grayscale = tf.concat(test_grayscale, axis=0)
            test_color = tf.concat(test_color, axis=0)
            
            # Generate images
            generated_images = generator(test_grayscale, training=False)
            
            # Calculate metrics
            ssim_score, psnr_score = calculate_metrics(test_color, generated_images)
            print(f"Epoch {epoch+1} - SSIM: {ssim_score:.4f}, PSNR: {psnr_score:.4f}")
            
            # Log to metrics CSV
            with open(metrics_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, ssim_score, psnr_score])
            
            # Save generated images
            generate_and_save_images(generator, epoch + 1, test_grayscale, output_dir=output_dir)
    
    # Plot losses
    plot_losses(losses_csv, output_dir=output_dir)
    
    # Plot metrics
    plot_metrics(metrics_csv, output_dir=output_dir)
    
    print("Training complete. Metrics and plots are saved in the 'output' directory.")

if __name__ == "__main__":
    main()
