import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class Thresholder():
    def __init__(self, threshold:int|None=None) -> None:
        if threshold is None:
            threshold = 1
        self.threshold = threshold

    def draw_horizontal_on_event(self, ax):
        def inner_func(event):
            if event.inaxes:
                if len(ax.lines) > 0:
                    ax.lines[0].remove()
                ax.axhline(y=event.ydata, color='r', linestyle='--', label='y={}'.format(event.ydata))
                self.threshold = event.ydata
                plt.draw()
        return inner_func

def visualize_errors(thresholder: Thresholder, cam_id, errors):
    ax = sns.barplot(errors)
    ax.figure.canvas.mpl_connect('button_press_event', thresholder.draw_horizontal_on_event(ax))
    ax.set_xlabel("Image Index")
    ax.set_ylabel("RMSE (px)")
    plt.title('Single Camera Calibration Errors: Cam ID {}'.format(cam_id))
    plt.show()

def visualize_stereo_errors(thresholder: Thresholder, errors):

    df = pd.DataFrame.from_dict({"Index": range(errors.shape[0]), "Left": errors[:, 0], "Right": errors[:, 1]})
    melted_df = pd.melt(df, id_vars=["Index"], value_vars=["Left", "Right"], value_name="Error", var_name="Camera")
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(melted_df, x="Index", y="Error", hue="Camera")
    ax.figure.canvas.mpl_connect('button_press_event', thresholder.draw_horizontal_on_event(ax))
    plt.title('Stereo Camera Calibration Errors')
    ax.set_xlabel("Image Index")
    ax.set_ylabel("RMSE (px)")
    plt.show()

