"""
Describe a training run by generating statistics and plots from a JSON log file.
"""

# imports
import argparse
import json
from pathlib import Path

# packages
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def load_log_data(log_path: Path) -> pl.DataFrame:
    """
    Load training log data from a JSON file.

    Args:
        log_path (Path): The path to the JSON log file.

    Returns:
        pl.DataFrame: The training log data.
    """
    # read the data
    with open(log_path, "rt", encoding="utf-8") as input_file:
        data = input_file.read()

    # parse the data
    parsed_data = [json.loads(line) for line in data.strip().split("\n")]

    # create a Polars DataFrame
    return pl.DataFrame(parsed_data)


def calculate_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate statistics for training log data.

    Args:
        df (pl.DataFrame): The training log data.

    Returns:
        pl.DataFrame: The statistics.
    """
    return df.select(
        [
            # loss stats
            pl.col("loss").mean().alias("mean_loss"),
            pl.col("loss").std().alias("std_loss"),
            pl.col("loss").min().alias("min_loss"),
            pl.col("loss").max().alias("max_loss"),
            # sample time stats
            pl.col("sample_time").mean().alias("mean_sample_time"),
            pl.col("sample_time").std().alias("std_sample_time"),
            pl.col("sample_time").min().alias("min_sample_time"),
            pl.col("sample_time").max().alias("max_sample_time"),
            # forward time stats
            pl.col("forward_time").mean().alias("mean_forward_time"),
            pl.col("forward_time").std().alias("std_forward_time"),
            pl.col("forward_time").min().alias("min_forward_time"),
            pl.col("forward_time").max().alias("max_forward_time"),
            # backward time stats
            pl.col("backward_time").mean().alias("mean_backward_time"),
            pl.col("backward_time").std().alias("std_backward_time"),
            pl.col("backward_time").min().alias("min_backward_time"),
            pl.col("backward_time").max().alias("max_backward_time"),
            # optimizer time stats
            pl.col("optimizer_time").mean().alias("mean_optimizer_time"),
            pl.col("optimizer_time").std().alias("std_optimizer_time"),
            pl.col("optimizer_time").min().alias("min_optimizer_time"),
            pl.col("optimizer_time").max().alias("max_optimizer_time"),
            # step time stats
            pl.col("step_time").mean().alias("mean_step_time"),
            pl.col("step_time").std().alias("std_step_time"),
            pl.col("step_time").min().alias("min_step_time"),
            pl.col("step_time").max().alias("max_step_time"),
        ]
    )


def plot_loss_by_step(df: pl.DataFrame, x_min: int = 1000) -> Path:
    """
    Plot the loss by step.

    Args:
        df (pl.DataFrame): The training log data.
        x_min (int): The minimum x-axis value to plot.

    Returns:
        Path: The path to the plot.
    """
    # set up the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # plot loss by step with 50% transparency
    sns.lineplot(x="step", y="loss", data=df.to_pandas(), alpha=0.25)
    plt.title("Loss by Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # now get the 100-step moving average loss and plot it
    loss_data = df.select(["step", "loss"]).to_pandas()
    loss_data["rolling_mean"] = loss_data["loss"].rolling(1000).mean()
    # percentiles
    loss_data["rolling_max"] = loss_data["loss"].rolling(1000).quantile(0.99)
    loss_data["rolling_min"] = loss_data["loss"].rolling(1000).quantile(0.01)
    sns.lineplot(x="step", y="rolling_mean", data=loss_data, color="red", alpha=0.9)
    sns.lineplot(x="step", y="rolling_max", data=loss_data, color="black", alpha=0.25)
    sns.lineplot(x="step", y="rolling_min", data=loss_data, color="black", alpha=0.25)

    # check if we have at least x_min steps
    if loss_data["step"].iloc[-1] > x_min:
        plt.xlim(x_min, loss_data["step"].iloc[-1])

    # make it log log via axes
    plt.yscale("log")
    plt.xscale("log")

    # add a horizontal line and label for the final moving average value
    final_avg = loss_data["rolling_mean"].iloc[-1]
    plt.axhline(final_avg, color="black", linestyle="--")
    plt.text(
        int(loss_data["step"].iloc[-1] * 1.01),
        final_avg + 0.1,
        f"{final_avg:.2f}",
        color="black",
    )

    # add min loss value line
    min_loss = loss_data["loss"].min()
    plt.axhline(min_loss, color="green", linestyle="--")
    plt.text(
        int(loss_data["step"].iloc[-1] * 1.01),
        min_loss,
        f"{min_loss:.2f}",
    )

    # save the plot
    plot_path = Path("loss_by_step.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def plot_step_time_components(df: pl.DataFrame, y_max_qt: float = 0.99) -> Path:
    """
    Plot the step time components.

    Args:
        df (pl.DataFrame): The training log data.
        y_max_qt (float): The quantile to use for the y-axis max value.

    Returns:
        Path: The path to the plot.
    """
    # set up the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # plot step time components
    step_time_data = df.select(
        ["step", "sample_time", "forward_time", "backward_time", "optimizer_time"]
    ).to_pandas()
    step_time_data.set_index("step", inplace=True)
    step_time_data.plot(kind="area", stacked=True, ax=plt.gca())
    plt.title("Step Time Components")
    plt.xlabel("Step")
    plt.ylabel("Time (seconds)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # set the y max to the quantile value of the sums plus 10% headspace
    qt_value = step_time_data.sum(axis=1).quantile(y_max_qt)
    plt.ylim(0, qt_value * 1.1)

    # save the plot
    plot_path = Path("step_time_components.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def plot_learning_rate_loss(df: pl.DataFrame) -> Path:
    """
    Plot the learning rate and loss.

    Args:
        df (pl.DataFrame): The training log data.

    Returns:
        Path: The path to the plot.
    """
    # set up the plotting style
    sns.set_style("whitegrid")

    # two panels
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)

    # first, plot the time series of learning rate by step
    sns.lineplot(x="step", y="lr", data=df.to_pandas())
    plt.title("Learning Rate by Step")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")

    # second, plot the scatter plot of learning rate vs loss with step as color
    plt.subplot(2, 1, 2)

    # scatter plot of learning rate vs loss with step as color
    sns.scatterplot(
        x="lr",
        y="loss_diff",
        hue="step",
        alpha=0.5,
        data=df.with_columns(loss_diff=pl.col("loss").diff()).to_pandas(),
    )
    plt.title("Learning Rate and Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")

    # save the plot
    plot_path = Path("learning_rate_loss.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Describe a training run by generating statistics and plots from a JSON log file."
    )
    arg_parser.add_argument("log_path", type=Path, help="Path to the JSON log file.")
    args = arg_parser.parse_args()

    # load the log data
    log_data = load_log_data(args.log_path)

    # calculate statistics
    statistics = calculate_statistics(log_data)
    print("Statistics:")

    # hide the column types
    pl.Config.set_tbl_formatting("UTF8_FULL")
    pl.Config.set_tbl_hide_column_data_types(True)
    print(statistics)

    # plot loss by step
    loss_by_step_plot = plot_loss_by_step(log_data)
    print(f"Loss by Step Plot: {loss_by_step_plot}")

    # plot step time components
    step_time_components_plot = plot_step_time_components(log_data)
    print(f"Step Time Components Plot: {step_time_components_plot}")

    # plot learning rate and loss
    lr_loss_plot = plot_learning_rate_loss(log_data)
    print(f"Learning Rate and Loss Plot: {lr_loss_plot}")
