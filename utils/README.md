# Utility scripts

This folder and this `README.md` contain some utility scripts.

## Get PNGs out of PDFs

```bash
convert *.pdf *.png
```

## Create GIFs

Create animations from the content of `00{6,7}/PDFs` into `anim/`:

```bash
for p in 006/PDFs/*; do
    g=anim/${p##*/}
    convert -delay 100 $p ${p/6/7} ${g/pdf/gif}
done
```

`delay` is expressed in `10`ms.

## Plotting loss functions

To plot the (*MSE*, *CE* and *rpl MSE*) cost functions interactively while training run `./show_error.plt -i`.
To statically plot them, after training, run `./show_error.plt`.
