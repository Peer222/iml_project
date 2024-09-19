from enum import Enum
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re
import torchvision
from torch import nn

from src.embedding_module import ResNetEmbeddingModule
from src.relation_module import RelationModuleFewShot
from src.dataset import CUBDataset
from src.util import load_model_state

from sklearn.metrics import confusion_matrix


class Color(Enum):
    BLACK = (0, 0, 0)
    ORANGE = (217 / 256, 130 / 256, 30 / 256)
    YELLOW = (227 / 256, 193 / 256, 0)
    BLUE = (55 / 256, 88 / 256, 136 / 256)
    GREY = (180 / 256, 180 / 256, 180 / 256)
    RED = (161 / 256, 34 / 256, 0)
    GREEN = (0, 124 / 256, 6 / 256)
    LIGHT_GREY = (240 / 256, 240 / 256, 240 / 256)


COLORS: list = [color.value for color in Color]


def plot(plt, file: Path | None = None) -> None:
    """Saves or shows given plot.

    Args:
        plt (pyplot): Plot
        file (Path, optional): Path of saved plot. If None the plot is only shown. Defaults to None.
    """
    # plt.tight_layout()
    if not file:
        plt.show()
    else:
        if not file.parent.is_dir():
            file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=file, dpi=300, bbox_inches="tight")
        plt.close()


def show_multiline_plot(
    data_series: pd.DataFrame,
    colors: list[Color] | None,
    x_label: str,
    y_label: str = "",
    y_labels: list[str] = [],
    filename: Path | None = None,
    hue: str | None = None,
    errorbar: tuple[str, int] | None = ("ci", 95),
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plots multiple line plots into single axes.

    Args:
        data_series (pd.DataFrame): errorbars are plotted if multiple values exist per x.
        colors (list[Color]): List of colors for each line plot.
        x_label (str): x_label determines x column and is used as x axis label.
        y_label (str): Label of the y axis. Defaults to "".
        y_labels (list[str]): If data_series is of type pd.Dataframe, labels provide the used column names.
        filename (Path, optional): Name of the file in that the plot is saved. If None plot is shown instead of saved. Defaults to None.
        hue (str, optional): Column name that is used for categorization. If hue is set default seaborn colors are used.
        errorbar (tuple[str, int]): If not None errorbars are computed. Defaults to ("ci", 95).
        ylim (tuple[float, float] | None): Optional limits of y axis. Defaults to None.
    """
    assert len(y_labels) > 0

    _, ax = plt.subplots(figsize=(14, 7))

    if hue:
        sns.lineplot(
            data=data_series,
            x=x_label,
            y=y_labels[0],
            hue=hue,
            ax=ax,
            errorbar=errorbar,
            dashes=False,
        )
    else:
        sns.lineplot(
            data=data_series.set_index("step")[y_labels],
            c=colors,
            ax=ax,
            errorbar=errorbar,
            dashes=False,
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.ylim(ylim)

    ax.grid(True, color=Color.LIGHT_GREY.value)
    sns.despine(left=True, bottom=True, right=True, top=True)

    plot(plt, filename)
    return


def open_train_dataframes(
    filepath: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Opens loss and accuracy dataframes from json training log file.
    log file format: {losses: {...}, accuracies: {...}, best_i: int}

    Args:
        filepath (Path | str): filepath to json log file

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of found dataframes in json training log file (losses, accuracies).
    """
    with open(filepath, "r") as f:
        content: dict = json.load(f)

    # -1 removes "best_i" which we do not need here
    return tuple(pd.DataFrame(content[key]) for key in list(content.keys())[:-1])


def open_test_dataframe(
    resultpath: Path | str,
) -> dict[str, pd.DataFrame]:
    """Open test dataframe from path
    log file format: {losses: {...}, accuracies: {...}, best_i: int}

    Args:
        filepath (Path | str): filepath to json

    Returns:
        dict with keys for different data and values are merged dataframes
    """

    def load_json(path: str | Path) -> dict:
        """ Lopad json file for dataframe

        Args:
            path: path to json

        Returns:
            json as dict
        """
        with open(path, "r") as f:
            return json.load(f)

    contents_per_dataset = [
        (load_json(Path(path) / "args.json"), load_json(Path(path) / "log.json"))
        for path in glob.glob(str(Path(resultpath) / "eval_*"))
    ]

    all_dataset_frames = []
    for args, contents in contents_per_dataset:
        dataset_frames = {
            key: pd.DataFrame(contents[key]) for key in list(contents.keys())
        }
        for frame in dataset_frames.values():
            frame["dataset"] = args["dataset"]
            frame["k_shot"] = args["num_fewshot_instances"]
            frame["model_mode"] = re.search(
                r"^.*\d+way\d+shot_(.*)", args["result_dir"]
            ).group(1)
        all_dataset_frames.append(dataset_frames)

    merged_dataframes = {}
    for key in all_dataset_frames[0].keys():
        merged_dataframes[key] = pd.concat(
            [frames[key] for frames in all_dataset_frames]
        ).reset_index(drop=True)

    return merged_dataframes


def open_test_dataframes(resultpaths: list[Path | str]) -> dict[str, pd.DataFrame]:
    """Open all test data from the paths and merge them together

    Args:
        resultpaths: result paths of different runs

    Returns:
        dict with keys for different data and values are merged dataframes
    """
    all_frames = [open_test_dataframe(path) for path in resultpaths]
    merged_frames = {}
    for key in all_frames[0].keys():
        merged_frames[key] = pd.concat(
            [frames[key] for frames in all_frames]
        ).reset_index(drop=True)
    return merged_frames


def plot_curves(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    filepath: str | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Plot loss/accuracy curves of training and validation data.

    Args:
        train_df (pd.Dataframe): Dataframe of logged training loss/accuracy with columns {data, step}
        val_df (pd.DataFrame): Dataframe of logged validation loss/accuracy with columns {data, step}
        filepath (str | None, optional): Path for saving plot. Defaults to None.
        ylim (tuple[float, float] | None): Optional limits of y axis. Defaults to None.
    """
    train_df["mode"] = "training"
    val_df["mode"] = "validation"
    df = pd.concat([train_df, val_df])
    show_multiline_plot(
        df,
        colors=None,
        x_label="step",
        y_label=df.columns[0],
        y_labels=[df.columns[0]],
        filename=filepath,
        hue="mode",
        errorbar=None,
        ylim=ylim,
    )


def plot_training_progress(filepath: Path | str):
    """ Plots that take full training run into account

    Args:
        filepath: path to log.json
    """
    train_loss_df, train_acc_df, val_loss_df, val_acc_df = open_train_dataframes(
        filepath
    )
    if isinstance(filepath, str):
        filepath = Path(filepath)

    plotpath = filepath.parent / "plots"
    plotpath.mkdir(exist_ok=True)

    train_loss_df = train_loss_df.rolling(1000, min_periods=1).mean()
    train_acc_df = train_acc_df.rolling(1000, min_periods=1).mean()

    # plot losses
    plot_curves(
        train_loss_df[["pred", "step"]].rename({"pred": "prediction loss"}, axis=1),
        val_loss_df[["pred", "step"]].rename({"pred": "prediction loss"}, axis=1),
        plotpath / "prediction_loss",
    )
    plot_curves(
        train_loss_df[["pred_lrp", "step"]].rename(
            {"pred_lrp": "lrp-guided prediction loss"}, axis=1
        ),
        val_loss_df[["pred_lrp", "step"]].rename(
            {"pred_lrp": "lrp-guided prediction loss"}, axis=1
        ),
        plotpath / "lrp_guided_prediction_loss",
    )
    plot_curves(
        train_loss_df[["total", "step"]].rename({"total": "total loss"}, axis=1),
        val_loss_df[["total", "step"]].rename({"total": "total loss"}, axis=1),
        plotpath / "total_loss",
    )

    # plot accuracy
    plot_curves(
        train_acc_df[["pred", "step"]].rename({"pred": "accuracy"}, axis=1),
        val_acc_df[["pred", "step"]].rename({"pred": "accuracy"}, axis=1),
        plotpath / "accuracy",
        ylim=(0.0, 1.05),
    )


def plot_evaluation(result_dirs: list[Path | str]) -> None:
    """ Plot all evaluation plots

    Args:
        result_dirs: directories (for different models) to use for results
    """

    def fix_cub_labels(labels: list[str]) -> list[str]:
        """ More beautiful labels from cub dataset

        Args:
            labels: labels out of cub dataset

        Returns:
            better labels
        """
        return [re.sub(r"\d+\.", "", label).replace("_", " ") for label in labels]

    dfs = open_test_dataframes(result_dirs)

    # Properly rename stuff in dataframe for plotting
    for df in dfs.values():
        df["dataset_names"] = (
            df["dataset"].str.replace("Dataset", "").replace("Cifar10", "CIFAR10")
        )
        mode_mapper = {
            "ablation": "Ablation",
            "baseline": "Baseline",
            "lrp_paper": "LRP - Paper",
            "lrp_variation": "LRP - Variation",
        }
        df["Mode"] = df["model_mode"].apply(lambda x: mode_mapper[x])
        df.rename(
            {"pred": "Accuracy", "dataset_names": "Dataset"}, axis=1, inplace=True
        )

    plot_path = Path(result_dirs[0]).parent / "test_plots"

    plot_path.mkdir(exist_ok=True)

    # boxplots for accuracy scores
    for k in dfs["accuracies"]["k_shot"].unique():
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(
            ax=ax,
            data=dfs["accuracies"][dfs["accuracies"]["k_shot"] == k],
            x="Accuracy",
            y="Dataset",
            hue="Mode",
        )
        ax.set_title("Accuracy over 2000 episodes with Batchsize 16")
        sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
        plot(plt, plot_path / f"5way{k}shot_accuracy_boxplot")

    # Confusion Matrix Plot for accuracy for 5-way 5-shot setting on CUB
    confusion_matrix_df = dfs["confusion_matrix"]
    confusion_matrix_df = confusion_matrix_df[
        (confusion_matrix_df["Dataset"] == "CUB") & (confusion_matrix_df["k_shot"] == 5)
    ].copy()

    def row_pred_indices_to_labels(row):
        return [row["labels"][i] for i in row["Accuracy"]]

    def row_gt_indices_to_labels(row):
        return [row["labels"][i] for i in row["ground_truth"]]

    confusion_matrix_df["pred_label"] = confusion_matrix_df.apply(
        row_pred_indices_to_labels, axis=1
    )
    confusion_matrix_df["ground_truth_label"] = confusion_matrix_df.apply(
        row_gt_indices_to_labels, axis=1
    )

    pred = confusion_matrix_df["pred_label"].sum()
    ground_truth = confusion_matrix_df["ground_truth_label"].sum()
    labels = sorted(confusion_matrix_df["labels"].iloc[0])

    cm = confusion_matrix(pred, ground_truth, labels=labels)
    cm = cm / cm.sum(axis=0)
    df_cm = pd.DataFrame(
        cm, index=fix_cub_labels(labels), columns=fix_cub_labels(labels)
    )

    plt.subplots(figsize=(7, 7))
    ax = sns.heatmap(
        df_cm,
        annot=True,
        cmap=sns.color_palette("light:#5A9", as_cmap=True),
        linewidths="1",
        fmt=".2f",
    )
    ax.set(xlabel="True label", ylabel="Predicted label")
    ax.set_title("5-way 5-shot LRP model")
    ax.set_xticklabels(fix_cub_labels(labels), rotation=45, ha="right")
    ax.set_yticklabels(fix_cub_labels(labels), rotation=45, va="top")

    plot(plt=plt, file=plot_path / "confusion_matrix.png")

    # Create visualization of supportset and q matrix using real network output
    resnet = ResNetEmbeddingModule()
    relationnet = RelationModuleFewShot(
        padding=1, in_channels=1024, hidden_channels=512, conv_out_channels=512
    )
    regex = re.compile(r"^.*lrp.*paper.*$")
    lrp_paper_dir = list(filter(lambda x: regex.match(str(x)), result_dirs))[0]
    relationnet = load_model_state(lrp_paper_dir, relationnet)
    dataset = CUBDataset(
        torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: x.convert("RGB")),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        ),
        mode="test",
        k_shot=5,
        n_way=5,
    )
    supportset, (query_images, labels) = dataset.sample_batch(3)
    supportset_embeddings = resnet.get_supportset_embeddings(supportset)
    query_images_embeddings = resnet.forward(query_images)

    relationnet.update_supportset(supportset_embeddings)
    scores = relationnet.forward(query_images_embeddings)
    class_probs = nn.Softmax(dim=1)(scores)

    fig, ax = plt.subplots(figsize=(14, 7))
    labels = [re.sub(r"\d+\.", "", label) for label in labels]
    sns.heatmap(
        ax=ax,
        data=pd.DataFrame(
            class_probs.detach().numpy(),
            index=fix_cub_labels(labels),
            columns=fix_cub_labels(relationnet.labels),
        ),
        cmap=sns.color_palette("light:#5A9", as_cmap=True),
        annot=True,
        linewidths=1,
    )
    ax.set_ylabel("Queryset")
    ax.set_xlabel("Supportset")
    ax.set_title("Model Output Representation")
    ax.set_xticklabels(fix_cub_labels(relationnet.labels), rotation=45, ha="right")
    ax.set_yticklabels(fix_cub_labels(labels), rotation=45, va="top")
    plt.tight_layout()
    plot(plt, plot_path / "5way5shot_support_query_visual")

    return


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="poster", palette="Set2")

    parser = argparse.ArgumentParser(description="Script for visualization")

    parser.add_argument(
        "--result_dirs",
        required=True,
        type=Path,
        help="Path to result directory with stored log data.",
        nargs="*",
    )

    args = parser.parse_args()

    for result_dir in args.result_dirs:
        plot_training_progress(result_dir / "log.json")
    plot_evaluation(args.result_dirs)
