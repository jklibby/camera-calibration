import matplotlib.pyplot as plt
import seaborn as sns

def visualize_errors(cam_id, errors):
    sns.histplot(errors, kde=True)
    plt.title('Histogram of Single Camera Calibration: {}'.format(cam_id))
    plt.show()

def visalize_stereo_errors(errors):
    x = errors[:, 0]
    y = errors[:, 1]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(x, kde=True)
    plt.title('Left Camera Errors')

    plt.subplot(1, 2, 2)
    sns.histplot(y, kde=True)
    plt.title('Right Camera Errors')

    plt.tight_layout()
    plt.show()
