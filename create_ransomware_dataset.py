import pandas as pd
import numpy as np
import os

def create_ransomware_dataset(output_path='data/PRATIRAKSHA_ransomware_dataset.csv'):
    os.makedirs('data', exist_ok=True)
    num_samples = 50000
    features = [f'feat_{i}' for i in range(1, 51)]
    attack_labels = ['Benign', 'Ransomware', 'Cryptolocker', 'WannaCry', 'Locky']
    data = []

    for _ in range(num_samples):
        label = np.random.choice(attack_labels, p=[0.6, 0.2, 0.08, 0.08, 0.04])
        if label == 'Benign':
            row = np.random.normal(loc=0, scale=1, size=50)
        else:
            row = np.random.normal(loc=2, scale=1.5, size=50)
        data.append(list(row) + [label])
    df = pd.DataFrame(data, columns=features + ['Label'])
    df.to_csv(output_path, index=False)
    print(f"🦠 Ransomware dataset created at {output_path}")

if __name__ == '__main__':
    create_ransomware_dataset()