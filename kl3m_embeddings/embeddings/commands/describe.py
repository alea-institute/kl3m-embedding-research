"""
Describe a training run by generating statistics and plots from a JSON log file.

Example log record:
{"step": 426, "epoch": 1, "lr": 9.075e-05, "reduced_dim": 16, "task": "mlm", "num_samples": 64, "num_identifiers": 4, "num_tokens": 8100, "samples_by_dataset": {"govinfo": 22, "dockets": 21, "fdlp": 21}, "tokens_by_dataset": {"govinfo": 2816, "dockets": 2688, "fdlp": 2596}, "sample_time": 0.02536940574645996, "loss": 6.867125988006592, "forward_time": 0.014142751693725586, "backward_time": 0.004799365997314453, "optimizer_time": 0.0006134510040283203, "step_time": 0.04641866683959961, "time": "2024-10-20T09:12:13.891810"}

Example eval record:
{"step": 19900, "mean": 3.8377096504606305, "median": 4.7862560749053955, "std": 2.825638274971136, "min": 0.05165662243962288, "p5": 0.07757293790578842, "p95": 7.967259349822998, "max": 9.941807746887207, "num_samples": 1000}

"""

# imports
import argparse
import json
from collections import Counter
from pathlib import Path

# packages
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def load_log_data(log_path: Path) -> tuple[pl.DataFrame, Counter]:
    """
    Load training log data from a JSON file.

    Args:
        log_path (Path): The path to the JSON log file.

    Returns:
        pl.DataFrame: The training log data.
    """
    # read the data
    parsed_data = []
    dataset_counts: Counter[str] = Counter()
    with open(log_path, "rt", encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            dataset_counts.update(record.pop("datasets", {}))
            parsed_data.append(record)

    # create a Polars DataFrame
    return pl.DataFrame(parsed_data), dataset_counts


def load_eval_data(eval_path: Path) -> pl.DataFrame:
    """
    Load evaluation data from a JSON file.

    Args:
        eval_path (Path): The path to the JSON evaluation file.

    Returns:
        pl.DataFrame: The evaluation data.
    """
    # read the data
    parsed_data = []
    with open(eval_path, "rt", encoding="utf-8") as input_file:
        for line in input_file:
            parsed_data.append(json.loads(line))

    # create a Polars DataFrame
    return pl.DataFrame(parsed_data)


def calculate_log_statistics(log_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate statistics for training log data.

    Args:
        log_df (pl.DataFrame): The training log data.

    Returns:
        pl.DataFrame: The statistics.
    """
    # get the final loss and eval loss
    return log_df.select(
        [
            # total steps is last value
            pl.col("step").last().alias("total_steps"),
            # last loss
            pl.col("loss").last().alias("final_loss"),
            # trailing 100 step loss mean
            pl.col("loss").tail(100).mean().alias("last_100_loss_mean"),
            # loss stats
            pl.col("loss").mean().alias("mean_loss"),
            pl.col("loss").std().alias("std_loss"),
            pl.col("loss").min().alias("min_loss"),
            pl.col("loss").max().alias("max_loss"),
            # sample time stats
            pl.col("sample_time").mean().alias("mean_sample_time"),
            # pl.col("sample_time").std().alias("std_sample_time"),
            # pl.col("sample_time").min().alias("min_sample_time"),
            # pl.col("sample_time").max().alias("max_sample_time"),
            # forward time stats
            pl.col("forward_time").mean().alias("mean_forward_time"),
            # pl.col("forward_time").std().alias("std_forward_time"),
            # pl.col("forward_time").min().alias("min_forward_time"),
            # pl.col("forward_time").max().alias("max_forward_time"),
            # backward time stats
            pl.col("backward_time").mean().alias("mean_backward_time"),
            # pl.col("backward_time").std().alias("std_backward_time"),
            # pl.col("backward_time").min().alias("min_backward_time"),
            # pl.col("backward_time").max().alias("max_backward_time"),
            # optimizer time stats
            pl.col("optimizer_time").mean().alias("mean_optimizer_time"),
            # pl.col("optimizer_time").std().alias("std_optimizer_time"),
            # pl.col("optimizer_time").min().alias("min_optimizer_time"),
            # pl.col("optimizer_time").max().alias("max_optimizer_time"),
            # step time stats
            pl.col("step_time").mean().alias("mean_step_time"),
            # pl.col("step_time").std().alias("std_step_time"),
            # pl.col("step_time").min().alias("min_step_time"),
            # pl.col("step_time").max().alias("max_step_time"),
        ]
    )


def calculate_eval_statistics(eval_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate statistics for evaluation data.

    Args:
        eval_df (pl.DataFrame): The evaluation data.

    Returns:
        pl.DataFrame: The statistics.
    """
    # get the final loss and eval loss
    return eval_df.select(
        [
            # total steps is last value
            pl.col("step").last().alias("total_steps"),
            # trend over last 10 evals
            pl.col("mean").tail(10).mean().alias("last_10_mean"),
            pl.col("median").tail(10).mean().alias("last_10_median"),
            # trend
            pl.col("median").tail(10).diff().mean().alias("last_10_median_diff_mean"),
            # loss stats
            pl.col("mean").mean().alias("mean_loss"),
            pl.col("mean").std().alias("std_loss"),
            pl.col("mean").min().alias("min_loss"),
            pl.col("mean").max().alias("max_loss"),
            # loss stats
            pl.col("median").mean().alias("mean_median"),
            pl.col("median").std().alias("std_median"),
            pl.col("median").min().alias("min_median"),
            pl.col("median").max().alias("max_median"),
            # p5 stats
            pl.col("p5").mean().alias("mean_p5"),
            pl.col("p5").std().alias("std_p5"),
            pl.col("p5").min().alias("min_p5"),
            pl.col("p5").max().alias("max_p5"),
            # p95 stats
            pl.col("p95").mean().alias("mean_p95"),
            pl.col("p95").std().alias("std_p95"),
            pl.col("p95").min().alias("min_p95"),
            pl.col("p95").max().alias("max_p95"),
        ]
    )


def plot_loss_by_step(
    log_df: pl.DataFrame, eval_df: pl.DataFrame, output_path: Path, x_min: int = 1000
) -> Path:
    """
    Plot the loss by step.

    Args:
        log_df (pl.DataFrame): The training log data.
        eval_df (pl.DataFrame): The evaluation data.
        output_path (Path): The output path for the plot.
        x_min (int): The minimum x-axis value to plot.

    Returns:
        Path: The path to the plot.
    """
    # set up the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # plot loss by step with 50% transparency
    # sns.lineplot(x="step", y="loss", data=df.to_pandas(), alpha=0.25)
    # plt.plot(log_df["step"], log_df["loss"], alpha=0.25)
    plt.title("Loss by Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # now get the 100-step moving average loss and plot it
    loss_data = log_df.select(["step", "loss"]).to_pandas()
    loss_data["rolling_mean"] = loss_data["loss"].rolling(100).mean()
    loss_data["rolling_max"] = (
        loss_data["loss"].rolling(100).quantile(0.95).rolling(100).mean()
    )
    loss_data["rolling_min"] = (
        loss_data["loss"].rolling(100).quantile(0.05).rolling(100).mean()
    )

    # plot the top and bottom lines as 50% black and fill between
    plt.fill_between(
        loss_data["step"],
        loss_data["rolling_min"],
        loss_data["rolling_max"],
        color="#c6c6b8",
        alpha=0.1,
    )
    plt.plot(loss_data["step"], loss_data["rolling_mean"], color="#d0aaaa", alpha=0.9)

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
    plt.axhline(min_loss, color="#aac6aa", linestyle="--")
    plt.text(
        int(loss_data["step"].iloc[-1] * 1.01),
        min_loss,
        f"{min_loss:.2f}",
    )

    # overplot the eval data if there are at least two datapoints
    if len(eval_df) > 1:
        # group by step to mean, then make sure they're ordered
        eval_loss_p5 = (
            eval_df.group_by("step").agg(pl.col("p5").mean().alias("mean")).sort("step")
        )
        eval_loss_mean = (
            eval_df.group_by("step")
            .agg(pl.col("mean").mean().alias("mean"))
            .sort("step")
        )
        eval_loss_p95 = (
            eval_df.group_by("step")
            .agg(pl.col("p95").mean().alias("mean"))
            .sort("step")
        )

        plt.fill_between(
            eval_loss_p5["step"],
            eval_loss_p5["mean"],
            eval_loss_p95["mean"],
            color="#aaaad0",
            alpha=0.1,
        )
        plt.scatter(
            eval_loss_mean["step"],
            eval_loss_mean["mean"],
            color="#aaaad0",
            alpha=0.9,
            marker="o",
            s=10,
        )

        # add last eval value with text annotation
        last_eval = eval_loss_mean["mean"][-1]
        plt.axhline(last_eval, color="#aaaad0", linestyle="--")
        plt.text(
            int(loss_data["step"].iloc[-1] * 1.01),
            last_eval,
            f"{last_eval:.2f}",
            color="#aaaad0",
        )

    # make it log log via axes
    # plt.yscale("log")
    # plt.xscale("log")

    # check if we have at least x_min steps
    if loss_data["step"].iloc[-1] > x_min:
        plt.xlim(x_min, loss_data["step"].iloc[-1])

        # get max loss value after x_min
        max_loss = loss_data[loss_data["step"] > x_min]["loss"].max()
        plt.ylim(0, max_loss * 1.1)
    else:
        plt.ylim(0, loss_data["loss"].max() * 1.1)

    # legend
    legend_names = ["Loss Range", "Loss MA", "Last Loss", "Min Loss"]
    if len(eval_df) > 1:
        legend_names.extend(["Eval Loss Range", "Eval Loss", "Last Eval"])
    plt.legend(legend_names, loc="upper center", ncols=len(legend_names))

    # save the plot
    plot_path = output_path / "loss_by_step.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def plot_step_time_components(
    df: pl.DataFrame, output_path: Path, y_max_qt: float = 0.99
) -> Path:
    """
    Plot the step time components.

    Args:
        df (pl.DataFrame): The training log data.
        output_path (Path): The output path for the plot.
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
    plt.legend(loc="upper left")

    # set the y max to the quantile value of the sums plus 10% headspace
    qt_value = step_time_data.sum(axis=1).quantile(y_max_qt)
    plt.ylim(0, qt_value * 1.25)

    # save the plot
    plot_path = output_path / "step_time_components.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def plot_learning_rate_loss(df: pl.DataFrame, output_path: Path) -> Path:
    """
    Plot the learning rate and loss.

    Args:
        df (pl.DataFrame): The training log data.
        output_path (Path): The output path for the plot.

    Returns:
        Path: The path to the plot.
    """
    # set up the plotting style
    sns.set_style("whitegrid")

    # two panels
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)

    # first, plot the time series of learning rate by step
    # sns.lineplot(x="step", y="lr", data=df.to_pandas())
    plt.plot(df["step"], df["lr"])
    plt.title("Learning Rate by Step")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")

    # second, plot the scatter plot of learning rate vs loss with step as color
    plt.subplot(2, 1, 2)

    # scatter plot of learning rate vs loss with step as color
    df = df.with_columns(loss_diff=pl.col("loss").diff())

    # plot the relationship with mean loss diff
    plt.scatter(
        df["lr"],
        df["loss"],
        c=df["step"],
        cmap="viridis",
        alpha=0.5,
        s=1,
    )

    plt.title("Learning Rate and Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")

    # save the plot
    plot_path = output_path / "learning_rate_loss.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


# histogram of samples by dataset id (datasets key/values in the log)
def plot_samples_by_dataset(df: pl.DataFrame, output_path: Path) -> Path:
    """
    Plot the number of samples by dataset.

    Args:
        df (pl.DataFrame): The training log data.
        output_path (Path): The output path for the plot.

    Returns:
        Path to plot file
    """
    # set up the plotting style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))

    # count samples by dataset id
    dataset_counts = df.group_by("dataset_id").agg(pl.count("sample_id").alias("count"))

    # plot the histogram
    sns.barplot(x="dataset_id", y="count", data=dataset_counts.to_pandas())
    plt.title("Samples by Dataset")
    plt.xlabel("Dataset ID")
    plt.ylabel("Sample Count")

    # save the plot
    plot_path = output_path / "samples_by_dataset.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Describe a training run by generating statistics and plots from a JSON log file."
    )
    arg_parser.add_argument("log_path", type=Path, help="Path to the JSON log file.")
    args = arg_parser.parse_args()

    # set up Polars configuration
    pl.Config.set_tbl_formatting("UTF8_FULL")
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_float_precision(4)

    # load the log data
    artifact_output_path = args.log_path.parent
    log_data, _ = load_log_data(args.log_path)
    eval_data = load_eval_data(artifact_output_path / "eval.jsonl")

    # calculate statistics
    statistics = calculate_log_statistics(log_data)
    print("Log Statistics:")
    print(statistics)

    try:
        eval_statistics = calculate_eval_statistics(eval_data)
        print("Eval Statistics:")
        print(eval_statistics)
    except Exception as e:  # pylint: disable=broad-except
        print(f"Error calculating eval statistics: {e}")

    # plot loss by step
    loss_by_step_plot = plot_loss_by_step(log_data, eval_data, artifact_output_path)
    print(f"Loss by Step Plot: {loss_by_step_plot}")

    # plot step time components
    step_time_components_plot = plot_step_time_components(
        log_data, artifact_output_path
    )
    print(f"Step Time Components Plot: {step_time_components_plot}")

    # plot learning rate and loss
    lr_loss_plot = plot_learning_rate_loss(log_data, artifact_output_path)
    print(f"Learning Rate and Loss Plot: {lr_loss_plot}")
