# Utility scripts

This folder and this `README.md` contain some very useful scripts.
The folder content is the following:

 - [`check_exp_diff.sh*`](check_exp_diff.sh): shows difference between experiments' training CLI arguments;
 - [`update_experiments.sh*`](update_experiments.sh): synchronises experiments across multiple nodes;
 - [`show_error.plt*`](show_error.plt): show current error with *gnuplot* by `awk`-ing `last/train.log`;
 - [`show_error_exp.plt*`](show_error_exp.plt): show losses for a specific experiment;
 - [`image_plot.py`](image_plot.py): saves intermediate training visual outputs to PDF (may be used by `main.py`);

## Script usage and arguments

### Compare CLI arguments

To compare the CLI arguments used in two different experiments run `./check_exp_diff.sh <base exp> <new exp>`.
The script will output the `--word-diff` between the calling arguments, and `git show` the newer experiment's `HEAD`.
Replace `<{base,new} exp>` with the corresponding three-digit identifier.

### Plotting current training losses

To plot the (*MSE*, *CE*, *replica MSE*, and *periodic CE*) cost functions interactively for the current training network run `./show_error.plt -i`.
To statically plot them, after training, run `./show_error.plt`.

### Displaying any experiment losses

If the experiment data is not on the current machine run `./update_experiments.sh -i` to iteratively sync the data every `5` seconds.
To view the losses iteratively run `./show_error_exp.plt <exp> -i`, where `<exp>` is for example `012`.
To run statically restrain to use the `-i` flag.

### Synchronising experiments

To sync experiments across multiple nodes run `./update_experiments.sh`.
If you whish to run it quietly (verbose is default), run `./update_experiments.sh -q`.
If you'd like to run it in a loop of `5` seconds, run `./update_experiments.sh -i`.

## Images manipulation scripts collection

### Get PNGs out of PDFs

```bash
convert *.pdf *.png
```

### Create GIFs

Create animations from the content of `00{6,7}/PDFs` into `anim/`:

```bash
for p in 006/PDFs/*; do
    g=anim/${p##*/}
    convert -delay 100 $p ${p/6/7} ${g/pdf/gif}
done
```

`delay` is expressed in `10`ms.