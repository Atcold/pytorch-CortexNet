# Utility scripts

This folder and this `README.md` contain some utility scripts.

## Images manipulation

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

## Plot loss functions

### Current training

To plot the (*MSE*, *CE*, *rpl MSE* and *per CE*) cost functions interactively for the current training network run `./show_error.plt -i`.
To statically plot them, after training, run `./show_error.plt`.

### Any experiment

If the experiment data is not on the current machine run `./update_experiments.sh -i` to iteratively sync the data every 5 seconds.
To view the losses interactively run `./show_error_exp.plt <exp> -i`, where `<exp>` is for example `012`.
To run statically restrain to use the `-i` flag.

## Sync experiments

To sync experiments across multiple nodes run `./update_experiments.sh`.
If you whish to run it quietly (verbose is default), run `./update_experiments.sh -q`.
If you'd like to run it in a loop of 5 seconds, run `./update_experiments.sh -i`.
