#!/usr/bin/env python3

import click

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c",
    "--bert-config",
    default="bert-large-uncased",
    help="Pretrained BERT configuration",
)
@click.option("-b", "--binary", is_flag=True, help="Use binary labels, ignore neutrals")
@click.option("-r", "--root", is_flag=True, help="Use only root nodes of SST")
@click.option(
    "-s", "--save", is_flag=True, help="Save the model files after every epoch"
)
def main(bert_config, binary, root, save):
    """Train BERT sentiment classifier."""
    from bert_sentiment.train import train

    train(binary=binary, root=root, bert=bert_config, save=save)


if __name__ == "__main__":
    main()
