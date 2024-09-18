from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# uses constrained_layout to adjust the plot layout for trace plot
# constrained_layout is similar to tight_layout,
# but uses a constraint solver to determine the size of axes that allows them to fit.
# ref: https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
plt.rcParams["figure.constrained_layout.use"] = True

TITLE_FONTSIZE = 32
LABEL_FONTSIZE = 24
TICKS_FONTSIZE = 16
OUTPUT_DIR = "output"


def format_date():
    myFmt = mdates.DateFormatter("%Y-%m-%d")
    plt.gca().xaxis.set_major_formatter(myFmt)


def plot_title(title: str):
    plt.suptitle(title, fontsize=TITLE_FONTSIZE)


def plot_xlabel(label: str):
    plt.xlabel(label, fontsize=LABEL_FONTSIZE)


def plot_ylabel(label: str):
    plt.ylabel(label, fontsize=LABEL_FONTSIZE)


def plot_ticks():
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)


def save_plot(filename: str, dpi: int = 1200, suffix: Literal["png", "svg"] = "png"):
    plt.savefig(f"{OUTPUT_DIR}/{filename}.{suffix}", dpi=dpi)
